import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

import memory_bank
import utils


def alternating_split(feature_sets):
    """
    Given a list of feature sets, return:
      - even-indexed sets as memory bank images
      - odd-indexed sets as normal test images
    """
    memory_sets = feature_sets[::2]
    test_sets = feature_sets[1::2]
    return memory_sets, test_sets


def build_feature_sets_from_files(folder: Path, filenames, desc: str):
    """
    Load only the given list of filenames from 'folder',
    and convert them into FeatureSet objects using utils.build_feature_sets.

    Assumes each .pt file contains a tensor feature map.
    """
    pairs = []
    for fname in tqdm(filenames, desc=desc):
        fpath = folder / fname
        try:
            tensor = torch.load(fpath)
        except Exception as e:
            print(f"[WARNING] Could not load {fpath}: {e}")
            continue
        image_name = Path(fname).stem
        pairs.append((image_name, tensor))

    feature_sets = utils.build_feature_sets(pairs)
    print(f"[INFO] Built {len(feature_sets)} feature sets from {folder}")
    return feature_sets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balanced heterogeneous PatchCore: Copernicus <-> SpectralEarth7bands"
    )
    parser.add_argument(
        "--copernicus-features",
        type=str,
        required=True,
        help="Path to Copernicus Pretrain normal feature folder",
    )
    parser.add_argument(
        "--se7-features",
        type=str,
        required=True,
        help="Path to SpectralEarth 7 bands normal feature folder",
    )
    parser.add_argument(
        "--anomalous-features",
        type=str,
        required=True,
        help="Path to anomalous (ActiveFire) feature folder",
    )
    parser.add_argument(
        "--banks-dir",
        type=str,
        required=True,
        help="Directory where memory banks will be saved",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory where result CSVs will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PatchCore (e.g. 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.1,
        help="Sampling ratio for memory bank coreset (random method).",
    )
    return parser.parse_args()


def build_bank_and_score(
    bank_sets,
    normal_test_sets,
    anomalous_sets,
    bank_path: Path,
    result_csv: Path,
    device: str,
    sampling_ratio: float = 0.1,
):
    """
    Build a memory bank from bank_sets, compute anomaly scores on
    normal_test_sets + anomalous_sets, and save both image-level
    and patch-level CSVs.

    Coreset method is fixed to 'random' as requested.
    """
    print(f"\n[INFO] === Building bank and scoring: {result_csv.name} ===")
    print(f"[INFO] Memory bank size (images): {len(bank_sets)}")
    print(f"[INFO] Normal test size (images): {len(normal_test_sets)}")
    print(f"[INFO] Anomalous test size (images): {len(anomalous_sets)}")

    # ----- Build memory bank -----
    print(f"[INFO] Creating MemoryBank (sampling_ratio={sampling_ratio}, method='random')")
    mb = memory_bank.MemoryBank(
        sampling_ratio=sampling_ratio,
        coreset_method="random",
        device=device,
    )

    print("[INFO] Building memory bank from patches...")
    mb.build([fs.patches for fs in tqdm(bank_sets, desc="[TQDM] Building memory")])
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    mb.save(bank_path)
    print(f"[INFO] Memory bank saved to {bank_path} ({mb.memory.size(0)} patches)")

    # ----- Compute anomaly scores -----
    print("[INFO] Computing anomaly scores on full test set (normal + anomalous)...")
    full_test_sets = normal_test_sets + anomalous_sets
    image_scores, patch_scores = mb.compute_anomaly_scores(
        full_test_sets, reduction="mean"
    )

    # Precompute normal test names for labeling
    normal_names = {fs.image_name for fs in normal_test_sets}

    # ----- Save image-level CSV -----
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving image-level scores to: {result_csv}")
    with result_csv.open("w") as f:
        f.write("image_name,anomaly_score,label\n")
        for name, score in image_scores:
            label = 0 if name in normal_names else 1
            f.write(f"{name},{score:.6f},{label}\n")

    # ----- Save patch-level CSV -----
    patch_csv = result_csv.with_name(result_csv.stem + "_patches.csv")
    print(f"[INFO] Saving patch-level scores to: {patch_csv}")
    with patch_csv.open("w") as f:
        f.write("image_name,patch_index,patch_score\n")
        for img_name, scores_list in patch_scores.items():
            for idx, s in enumerate(scores_list):
                f.write(f"{img_name},{idx},{s:.6f}\n")

    print("[INFO] Done for:", result_csv.name)


def main():
    args = parse_args()

    cop_folder = Path(args.copernicus_features)
    se7_folder = Path(args.se7_features)
    anom_folder = Path(args.anomalous_features)

    banks_dir = Path(args.banks_dir)
    results_dir = Path(args.results_dir)

    # ---------------------------------------------------------
    # 1) Get file lists and determine balancing
    # ---------------------------------------------------------
    cop_files_all = sorted([f for f in cop_folder.iterdir() if f.suffix == ".pt"])
    se7_files_all = sorted([f for f in se7_folder.iterdir() if f.suffix == ".pt"])

    cop_fnames_all = [f.name for f in cop_files_all]
    se7_fnames_all = [f.name for f in se7_files_all]

    len_cop = len(cop_fnames_all)
    len_se7 = len(se7_fnames_all)

    print(f"[INFO] Copernicus .pt files: {len_cop}")
    print(f"[INFO] SpectralEarth7bands .pt files: {len_se7}")

    if len_cop == 0 or len_se7 == 0:
        raise RuntimeError("[ERROR] One of the normal folders is empty.")

    # target length = min of the two
    target_len = min(len_cop, len_se7)
    print(f"[INFO] Target balanced size: {target_len}")

    # Downsample Copernicus if needed
    if len_cop > target_len:
        step = len_cop // target_len
        if step < 1:
            step = 1
        indices = np.arange(0, step * target_len, step)
        indices = indices[:target_len]
        cop_fnames_balanced = [cop_fnames_all[i] for i in indices]
        print(
            f"[INFO] Downsampling Copernicus: {len_cop} -> {len(cop_fnames_balanced)} "
            f"using step={step}, last index={indices[-1]}"
        )
    else:
        cop_fnames_balanced = cop_fnames_all
        print("[INFO] Copernicus already at target size; no downsampling.")

    # Downsample SE7 if needed (in your case len_se7 == target_len, so no)
    if len_se7 > target_len:
        step = len_se7 // target_len
        if step < 1:
            step = 1
        indices = np.arange(0, step * target_len, step)
        indices = indices[:target_len]
        se7_fnames_balanced = [se7_fnames_all[i] for i in indices]
        print(
            f"[INFO] Downsampling SpectralEarth7bands: {len_se7} -> {len(se7_fnames_balanced)} "
            f"using step={step}, last index={indices[-1]}"
        )
    else:
        se7_fnames_balanced = se7_fnames_all
        print("[INFO] SpectralEarth7bands already at target size; no downsampling.")

    print(
        f"[INFO] After balancing: Copernicus={len(cop_fnames_balanced)}, "
        f"SpectralEarth7bands={len(se7_fnames_balanced)}"
    )

    # ---------------------------------------------------------
    # 2) Load only the balanced subsets as FeatureSets
    # ---------------------------------------------------------
    cop_sets_balanced = build_feature_sets_from_files(
        cop_folder, cop_fnames_balanced, desc="Loading Copernicus (balanced)"
    )
    se7_sets_balanced = build_feature_sets_from_files(
        se7_folder, se7_fnames_balanced, desc="Loading SpectralEarth7bands (balanced)"
    )

    # Alternating split
    cop_bank_sets, cop_test_sets = alternating_split(cop_sets_balanced)
    se7_bank_sets, se7_test_sets = alternating_split(se7_sets_balanced)

    print(
        f"[INFO] Copernicus -> bank: {len(cop_bank_sets)}, normal test: {len(cop_test_sets)}"
    )
    print(
        f"[INFO] SpectralEarth7bands -> bank: {len(se7_bank_sets)}, normal test: {len(se7_test_sets)}"
    )

    # ---------------------------------------------------------
    # 3) Load anomalous features once
    # ---------------------------------------------------------
    print("\n[INFO] Loading anomalous features (ActiveFire)...")
    anom_pairs = []
    anom_files = sorted([f for f in anom_folder.iterdir() if f.suffix == ".pt"])
    for fpath in tqdm(anom_files, desc="Loading anomalous"):
        try:
            tensor = torch.load(fpath)
        except Exception as e:
            print(f"[WARNING] Could not load {fpath}: {e}")
            continue
        image_name = fpath.stem
        anom_pairs.append((image_name, tensor))

    anomalous_sets = utils.build_feature_sets(anom_pairs)
    print(f"[INFO] Loaded {len(anomalous_sets)} anomalous feature sets.")

    # ---------------------------------------------------------
    # 4) Heterogeneous experiment A:
    #    Copernicus bank, SE7 normal test
    # ---------------------------------------------------------
    bankA_path = banks_dir / "copernicuspretrainbankspectralearth7bandsnormalbalanced_bank.pt"
    resultA_csv = results_dir / "copernicuspretrainbankspectralearth7bandsnormalbalanced.csv"

    build_bank_and_score(
        bank_sets=cop_bank_sets,
        normal_test_sets=se7_test_sets,
        anomalous_sets=anomalous_sets,
        bank_path=bankA_path,
        result_csv=resultA_csv,
        device=args.device,
        sampling_ratio=args.sampling_ratio,
    )

    # ---------------------------------------------------------
    # 5) Heterogeneous experiment B:
    #    SE7 bank, Copernicus normal test
    # ---------------------------------------------------------
    bankB_path = banks_dir / "spectralearth7bandsbankcopernicuspretrainnormalbalanced_bank.pt"
    resultB_csv = results_dir / "spectralearth7bandsbankcopernicuspretrainnormalbalanced.csv"

    build_bank_and_score(
        bank_sets=se7_bank_sets,
        normal_test_sets=cop_test_sets,
        anomalous_sets=anomalous_sets,
        bank_path=bankB_path,
        result_csv=resultB_csv,
        device=args.device,
        sampling_ratio=args.sampling_ratio,
    )

    print("\n[INFO] All balanced heterogeneous PatchCore runs completed.")


if __name__ == "__main__":
    main()

'''

python3 patchcore_heterogeneous_balanced.py --copernicus-features ../HyperfreeBackbone-ActiveFireNoThermal/Extracted\ Features/features\ copernicus\ pretrain  --se7-features ../HyperfreeBackbone-ActiveFireNoThermal/Extracted\ Features/features\ spectralearth7bands  --anomalous-features ../HyperfreeBackbone-ActiveFireNoThermal/Extracted\ Features/features\ anomalous\ set  --banks-dir ../../../data/datasets/PatchCore/banks  --results-dir ../../../data/datasets/PatchCore/results/mixed   --device cpu   --sampling-ratio 0.1

'''