import argparse
import os
from pathlib import Path
from tqdm import tqdm
import time
import torch
import memory_bank, utils

def alternating_split(features):
    even = features[::2]  # memory bank
    odd = features[1::2]  # test set
    return even, odd

def run_inference_with_external_bank(
    test_normal_folder: Path,
    anomalous_folder: Path,
    bank_path: Path,
    result_csv: Path,
    device: str,
):
    print(f"[INFO] Loading memory bank from: {bank_path}")
    if not bank_path.exists():
        raise FileNotFoundError(f"Memory bank file not found at: {bank_path}")
    bank = memory_bank.MemoryBank.load(bank_path, device=device)
    print(f"[INFO] Loaded memory bank with {bank.memory.size(0)} patches")

    print(f"[INFO] Loading test normal features from: {test_normal_folder}")
    test_normal_pairs = utils.load_feature_tensors(test_normal_folder)
    test_normal_sets = utils.build_feature_sets(test_normal_pairs)

    if test_normal_folder.as_posix().endswith("features copernicus pretrain"):
        print("[INFO] Using only one-third of normal images for CopernicusPretrain dataset.")
        test_normal_sets = test_normal_sets[::3]

    print("[INFO] Splitting normal features into memory and test sets...")
    #test_memory_sets, test_normal_sets = alternating_split(test_normal_sets)

    print(f"[INFO] Loading anomalous features from: {anomalous_folder}")
    anomalous_pairs = utils.load_feature_tensors(anomalous_folder)
    anomalous_sets = utils.build_feature_sets(anomalous_pairs)

    print("[INFO] Scoring anomaly on test set (normal + anomalous)...")
    full_test_set = test_normal_sets + anomalous_sets
    start = time.time()
    image_scores, patch_scores = bank.compute_anomaly_scores(full_test_set, reduction="mean")
    print(f"[TIMER] Anomaly scoring done in {time.time() - start:.2f}s")

    print(f"[INFO] Saving results to {result_csv}")
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    with result_csv.open("w") as f:
        f.write("image_name,anomaly_score,label\n")
        for name, score in image_scores:
            label = 0 if name in [fs.image_name for fs in test_normal_sets] else 1
            f.write(f"{name},{score:.6f},{label}\n")

    patch_csv = result_csv.with_name(result_csv.stem + "_patches.csv")
    print(f"[INFO] Saving patch-level results to {patch_csv}")
    with patch_csv.open("w") as f:
        f.write("image_name,patch_index,patch_score\n")
        for image_name, patch_list in patch_scores.items():
            for idx, score in enumerate(patch_list):
                f.write(f"{image_name},{idx},{score:.6f}\n")

    print("[INFO] Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-normal-features", type=str, required=True)
    parser.add_argument("--anomalous-features", type=str, required=True)
    parser.add_argument("--bank-path", type=str, required=True, help="Path to the prebuilt memory bank file")
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    run_inference_with_external_bank(
        test_normal_folder=Path(args.test_normal_features),
        anomalous_folder=Path(args.anomalous_features),
        bank_path=Path(args.bank_path),
        result_csv=Path(args.results_path),
        device=args.device,
    )


if __name__ == "__main__":
    main()


'''
python3 test_with_external_bank.py \
  --test-normal-features ../Extracted\ Features/features\ hyperseg \
  --anomalous-features ../Extracted\ Features/features\ anomalous\ set \
  --bank-path banks/copernicuspretrain_random_0.1.pt \
  --results-path results/cross_eval/cop2hyp.csv \
  --device cuda

python3 evaluate_test_set.py --test-normal-features ../HyperfreeBackbone-ActiveFireNoThermal/Extracted\ Features/features\ spectralearth\ 7\ bands --anomalous-features ../HyperfreeBackbone-ActiveFireNoThermal/Extracted\ Features/features\ anomalous\ set --bank-path ../../../data/datasets/PatchCore/banks/copernicuspretrain_random_0.1.pt --results-path ../../../data/datasets/PatchCore/results/mixed/copernicuspretrainbankspectralearth7bandsnormal.csv --device cuda

python3 evaluate_test_set.py --test-normal-features ../../../data/datasets/hyperfree/Extracted\ Features/features\ floga\ pre --anomalous-features ../../../data/datasets/hyperfree/Extracted\ Features/features\ floga\ post --bank-path ../../../data/datasets/PatchCore/banks/copernicuspretrain_random_0.1.pt --results-path ../../../data/datasets/PatchCore/results/floga/copernicuspretrainbankfloganormal.csv --device cuda

'''