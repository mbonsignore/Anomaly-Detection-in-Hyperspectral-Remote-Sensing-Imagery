#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

import memory_bank
import utils


def alternating_split(items):
    """
    Alternating split: even indices -> memory, odd indices -> test.
    """
    even = items[::2]  # memory bank
    odd = items[1::2]  # test set
    return even, odd


def load_label_map(labels_csv: Path) -> dict:
    """
    Load a mapping sample_name -> label from a CSV.

    Expected columns:
        sample_name,label

    sample_name should match the feature file stem (without .pt),
    e.g. FLOGA_2020_..._pre or ..._post.
    """
    label_map = {}
    with labels_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"sample_name", "label"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV {labels_csv} must contain columns {sorted(expected)}, "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            name = row["sample_name"]
            label = int(row["label"])
            label_map[name] = label

    num_anom = sum(1 for v in label_map.values() if v == 1)
    num_norm = len(label_map) - num_anom
    print(f"[INFO] Loaded labels from {labels_csv}: {num_norm} normal, {num_anom} anomalous.")
    return label_map


def build_and_score_pipeline_separate(
    normal_folder: Path,
    anomalous_folder: Path,
    bank_path: Path,
    result_csv: Path,
    sampling_ratio: float,
    coreset_method: str,
    device: str,
    dataset_name: str,
    load_existing: bool = False,
) -> None:
    """
    ORIGINAL MODE:
      - normal_folder: folder with normal features (.pt)
      - anomalous_folder: folder with anomalous features (.pt)
      - alternating split on normals: memory vs normal test
      - anomalies all go to test
    """

    print("[INFO] Listing normal feature files from:", normal_folder)
    normal_files = utils.list_feature_files(normal_folder)

    if dataset_name.lower() == "copernicuspretrain":
        print("[INFO] Using only one-third of normal images for CopernicusPretrain dataset.")
        normal_files = normal_files[::3]

    memory_files = normal_files[::2]
    normal_test_files = normal_files[1::2]

    if load_existing and bank_path.exists():
        print(f"[INFO] Loading existing memory bank from {bank_path}...")
        bank = memory_bank.MemoryBank.load(bank_path, device=device)
    else:
        print(f"[INFO] Building memory bank with {len(memory_files)} images...")
        bank = memory_bank.MemoryBank(
            sampling_ratio=sampling_ratio,
            coreset_method=coreset_method,
            device=device,
        )
        bank.build(list(utils.iter_patches_from_files(memory_files, device="cpu")))
        bank.save(bank_path)

    print("[INFO] Loading normal TEST features (from file list)...")
    normal_test_pairs = [(p.name, torch.load(p, map_location="cpu")) for p in tqdm(normal_test_files)]
    normal_test_sets = utils.build_feature_sets(normal_test_pairs)

    print("[INFO] Loading anomalous features from:", anomalous_folder)
    anomalous_pairs = utils.load_feature_tensors(anomalous_folder)
    anomalous_sets = utils.build_feature_sets(anomalous_pairs)

    print("[INFO] Scoring anomaly on test set (normal + anomalous)...")
    full_test_set = normal_test_sets + anomalous_sets
    image_scores, patch_scores = bank.compute_anomaly_scores(full_test_set, reduction="mean")

    # In this mode we infer labels from membership in the normal_test_files set
    normal_test_stems = {Path(p).stem for p in normal_test_files}

    print(f"[INFO] Saving results to {result_csv}")
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    with result_csv.open("w") as f:
        f.write("image_name,anomaly_score,label\n")
        for name, score in image_scores:
            stem = Path(name).stem
            label = 0 if stem in normal_test_stems else 1
            f.write(f"{name},{score:.6f},{label}\n")

    patch_csv = result_csv.with_name(result_csv.stem + "_patches.csv")
    print(f"[INFO] Saving patch-level results to {patch_csv}")
    with patch_csv.open("w") as f:
        f.write("image_name,patch_index,patch_score\n")
        for image_name, patch_list in patch_scores.items():
            for idx, score in enumerate(patch_list):
                f.write(f"{image_name},{idx},{score:.6f}\n")

    print("[INFO] Done (separate-folders mode).")


def build_and_score_pipeline_single(
    features_folder: Path,
    labels_csv: Path,
    bank_path: Path,
    result_csv: Path,
    sampling_ratio: float,
    coreset_method: str,
    device: str,
    dataset_name: str,
    load_existing: bool = False,
) -> None:
    """
    NEW MODE:
      - features_folder: one folder with both normal + anomalous .pt features
      - labels_csv: CSV with columns sample_name,label
      - uses labels to:
          * split normals/anomalies
          * alternating split only on normals → memory vs normal test
      - uses CSV labels when writing the results.
    """

    print("[INFO] Loading labels from CSV:", labels_csv)
    label_map = load_label_map(labels_csv)

    print("[INFO] Listing feature files from:", features_folder)
    all_files = utils.list_feature_files(features_folder)

    # Separate normal and anomalous according to CSV
    normal_files = []
    anomalous_files = []

    for p in all_files:
        stem = Path(p).stem
        if stem not in label_map:
            print(f"[WARN] Feature file {p} not found in label CSV; treating as normal (label=0).")
            lbl = 0
        else:
            lbl = label_map[stem]

        if lbl == 0:
            normal_files.append(p)
        else:
            anomalous_files.append(p)

    print(f"[INFO] Using {len(normal_files)} normal files and {len(anomalous_files)} anomalous files.")

    # Optional dataset-specific thinning (kept here for consistency)
    if dataset_name.lower() == "copernicuspretrain":
        print("[INFO] Using only one-third of normal images for CopernicusPretrain dataset.")
        normal_files = normal_files[::3]

    # Alternating split on NORMALS only
    memory_files, normal_test_files = alternating_split(normal_files)
    print(f"[INFO] Memory bank will use {len(memory_files)} normal images.")
    print(f"[INFO] Normal test set contains {len(normal_test_files)} normal images.")
    print(f"[INFO] Anomalous test set contains {len(anomalous_files)} anomalous images.")

    # Build / load memory bank
    if load_existing and bank_path.exists():
        print(f"[INFO] Loading existing memory bank from {bank_path}...")
        bank = memory_bank.MemoryBank.load(bank_path, device=device)
    else:
        print(f"[INFO] Building memory bank with {len(memory_files)} images...")
        bank = memory_bank.MemoryBank(
            sampling_ratio=sampling_ratio,
            coreset_method=coreset_method,
            device=device,
        )
        bank.build(list(utils.iter_patches_from_files(memory_files, device="cpu")))
        bank.save(bank_path)

    # Build normal test sets
    print("[INFO] Loading NORMAL test features (from file list)...")
    normal_test_pairs = [
        (p.name, torch.load(p, map_location="cpu")) for p in tqdm(normal_test_files, desc="Loading normal test")
    ]
    normal_test_sets = utils.build_feature_sets(normal_test_pairs)

    # Build anomalous test sets
    print("[INFO] Loading ANOMALOUS test features (from file list)...")
    anomalous_pairs = [
        (p.name, torch.load(p, map_location="cpu")) for p in tqdm(anomalous_files, desc="Loading anomalous test")
    ]
    anomalous_sets = utils.build_feature_sets(anomalous_pairs)

    # Concatenate and score
    print("[INFO] Scoring anomaly on test set (normal + anomalous)...")
    full_test_set = normal_test_sets + anomalous_sets
    image_scores, patch_scores = bank.compute_anomaly_scores(full_test_set, reduction="mean")

    print(f"[INFO] Saving results to {result_csv}")
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    # Use labels from CSV for final labels
    with result_csv.open("w") as f:
        f.write("image_name,anomaly_score,label\n")
        for name, score in image_scores:
            stem = Path(name).stem  # "xxx.pt" → "xxx"
            label = label_map.get(stem, 0)  # default to 0 if missing
            f.write(f"{name},{score:.6f},{label}\n")

    # Patch-level scores
    patch_csv = result_csv.with_name(result_csv.stem + "_patches.csv")
    print(f"[INFO] Saving patch-level results to {patch_csv}")
    with patch_csv.open("w") as f:
        f.write("image_name,patch_index,patch_score\n")
        for image_name, patch_list in patch_scores.items():
            for idx, score in enumerate(patch_list):
                f.write(f"{image_name},{idx},{score:.6f}\n")

    print("[INFO] Done (single-folder + CSV mode).")


def parse_args():
    parser = argparse.ArgumentParser()

    # OLD MODE ARGS
    parser.add_argument("--normal-features", type=str, help="Folder with normal feature .pt files.")
    parser.add_argument("--anomalous-features", type=str, help="Folder with anomalous feature .pt files.")

    # NEW MODE ARGS
    parser.add_argument(
        "--features-folder",
        type=str,
        help="Folder with both normal and anomalous feature .pt files (used with --labels-csv).",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        help="CSV with sample_name,label used to split normal and anomalous features.",
    )

    parser.add_argument("--bank-dir", type=str, default="banks")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--sampling-ratio", type=float, default=1.0)
    parser.add_argument("--coreset-method", choices=["random", "greedy"], default="random")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--load-bank",
        action="store_true",
        help="Load existing memory bank instead of rebuilding it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Decide which mode to use
    use_single_folder = args.features_folder is not None and args.labels_csv is not None

    if use_single_folder:
        print("[INFO] Running in SINGLE-FOLDER + CSV mode.")
        features_folder = Path(args.features_folder)
        labels_csv = Path(args.labels_csv)
        if not features_folder.is_dir():
            raise SystemExit(f"Features folder does not exist: {features_folder}")
        if not labels_csv.is_file():
            raise SystemExit(f"Labels CSV does not exist: {labels_csv}")
    else:
        print("[INFO] Running in SEPARATE-FOLDERS mode.")
        if args.normal_features is None or args.anomalous_features is None:
            raise SystemExit(
                "In separate-folders mode you must provide both --normal-features and --anomalous-features. "
                "Alternatively, use --features-folder and --labels-csv."
            )

    # Memory bank + results paths
    mem_bank_name = f"{args.dataset_name}_{args.coreset_method}_{args.sampling_ratio}.pt"
    bank_path = Path(args.bank_dir) / mem_bank_name

    results_dir = Path(args.results_root) / args.dataset_name / args.coreset_method
    result_csv = results_dir / f"{args.sampling_ratio}.csv"

    if use_single_folder:
        build_and_score_pipeline_single(
            features_folder=features_folder,
            labels_csv=labels_csv,
            bank_path=bank_path,
            result_csv=result_csv,
            sampling_ratio=args.sampling_ratio,
            coreset_method=args.coreset_method,
            device=args.device,
            dataset_name=args.dataset_name,
            load_existing=args.load_bank,
        )
    else:
        build_and_score_pipeline_separate(
            normal_folder=Path(args.normal_features),
            anomalous_folder=Path(args.anomalous_features),
            bank_path=bank_path,
            result_csv=result_csv,
            sampling_ratio=args.sampling_ratio,
            coreset_method=args.coreset_method,
            device=args.device,
            dataset_name=args.dataset_name,
            load_existing=args.load_bank,
        )


if __name__ == "__main__":
    main()

'''
    python3 full_patchcore_pipeline.py \
--normal-features ../Extracted\ Features/features\ hyperseg \
--anomalous-features ../Extracted\ Features/features\ anomalous\ set \
--bank-dir banks \
--results-root results \
--dataset-name hyperseg \
--sampling-ratio 0.1 \
--coreset-method greedy \
--load-bank

python3 full_patchcore_pipeline.py --normal-features ../../../data/datasets/Copernicus-FM/Extracted\ Features/features\ copernicus\ pretrain --anomalous-features ../../../data/datasets/Copernicus-FM/Extracted\ Features/features\ anomalous\ set --bank-dir ../../../data/datasets/PatchCore/banks/copernicusfm --results-root ../../../data/datasets/PatchCore/results/copernicusfm --dataset-name copernicuspretrain --sampling-ratio 0.1 --coreset-method random --load-bank 


python3 full_patchcore_pipeline.py --features-folder ../../../data/datasets/hyperfree/Extracted\ Features/finetuning/features\ floga\ test --labels-csv ../../../data/datasets/FLOGA/finetuning\ data/floga_splits/images_test.csv --bank-dir ../../../data/datasets/PatchCore/banks --results-root ../../../data/datasets/PatchCore/results/finetuning --dataset-name flogatestfinetuning --sampling-ratio 0.1 --coreset-method random 

'''