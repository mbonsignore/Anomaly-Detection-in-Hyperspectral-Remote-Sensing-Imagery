#!/usr/bin/env python3
"""
Compute statistics of distances between training samples
and the normal centroid, using the teacher cache:

    floga_teacher_cache_semisupervised.pt

We compute:
  - distances of NORMAL samples to the center
  - distances of ANOMALOUS samples to the center

Assumptions:
  - The cache file was created by train_finetune_copernicus_floga_semisupervised.py
    and contains:
        {"cache": {path: embedding}, "center": center_tensor}
  - The train CSV has columns: sample_name, folder, label
    with label=0 (normal), label=1 (anomalous).
"""

import argparse
import csv
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compute distances to centroid for normal and anomalous samples from teacher cache."
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=str(base_dir / "ckpt/floga_teacher_cache_semisupervised.pt"),
        help="Path to the teacher cache file (with 'cache' and 'center').",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(base_dir / "../Datasets/FLOGA/finetuning data/floga_splits/images_train_semisupervised.csv"),
        help="Path to the semi-supervised train CSV.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(base_dir / "../Datasets/FLOGA"),
        help="Root directory where FLOGA .pt files live.",
    )

    return parser.parse_args()


def compute_and_print_stats(dists: torch.Tensor, title: str) -> None:
    """Utility to print basic stats + quantiles for a distance vector."""
    if dists.numel() == 0:
        print(f"\n[RESULT] {title}: no samples.")
        return

    mean_dist = dists.mean().item()
    std_dist = dists.std().item()

    print(f"\n[RESULT] {title}:")
    print(f"  N samples       : {dists.numel()}")
    print(f"  mean distance   : {mean_dist:.4f}")
    print(f"  std  distance   : {std_dist:.4f}")

    for q in [0.05, 0.25, 0.50, 0.75, 0.90, 0.95]:
        val = torch.quantile(dists, q).item()
        print(f"  q={q:>4.2f} distance : {val:.4f}")

    # Optional: min/max, sometimes handy
    print(f"  min distance    : {dists.min().item():.4f}")
    print(f"  max distance    : {dists.max().item():.4f}")


def main() -> None:
    args = parse_args()

    cache_path = Path(args.cache_path)
    train_csv = Path(args.train_csv)
    data_root = Path(args.data_root)

    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    print(f"[INFO] Loading teacher cache from: {cache_path}")
    cache_data = torch.load(cache_path, map_location="cpu")

    # teacher_cache: dict[str, Tensor], center: Tensor[D]
    teacher_cache = cache_data["cache"]
    center = cache_data["center"].float()  # [D]
    print(f"[INFO] Loaded center with shape: {tuple(center.shape)}")
    print(f"[INFO] Cache entries: {len(teacher_cache)}")

    normal_paths = []
    anomaly_paths = []

    print(f"[INFO] Reading train CSV: {train_csv}")
    with train_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"sample_name", "folder", "label"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV {train_csv} must include columns {sorted(expected)}, "
                f"found {reader.fieldnames}"
            )

        for row in reader:
            label = int(row["label"])
            sample_name = row["sample_name"]
            folder = row["folder"]

            # Reconstruct the path exactly like FlogaCsvDataset.__getitem__
            path = data_root / folder / f"{sample_name}.pt"
            key = str(path)

            if key not in teacher_cache:
                # If needed, uncomment to debug missing keys:
                # print(f"[WARN] Path not in cache: {key} (label={label})")
                continue

            if label == 0:
                normal_paths.append(key)
            elif label == 1:
                anomaly_paths.append(key)

    if not normal_paths:
        raise RuntimeError("No NORMAL paths found in both CSV and cache.")
    if not anomaly_paths:
        raise RuntimeError("No ANOMALOUS paths found in both CSV and cache.")

    print(f"[INFO] Normal samples found in cache   : {len(normal_paths)}")
    print(f"[INFO] Anomalous samples found in cache: {len(anomaly_paths)}")

    # Stack embeddings and compute distances to center
    def get_dists(paths):
        embs = torch.stack([teacher_cache[p].float() for p in paths], dim=0)  # [N, D]
        if embs.ndim != 2:
            raise ValueError(f"Expected 2D embeddings [N,D], got {embs.shape}")
        c = center.view(-1)
        if embs.shape[1] != c.shape[0]:
            raise ValueError(
                f"Embedding dim ({embs.shape[1]}) != center dim ({c.shape[0]}). "
                "Check that you are using the correct cache file."
            )
        diffs = embs - c.unsqueeze(0)  # [N, D]
        return diffs.norm(dim=1)       # [N]

    normal_dists = get_dists(normal_paths)
    anomaly_dists = get_dists(anomaly_paths)

    # Print stats
    compute_and_print_stats(normal_dists, "NORMAL distances to center")
    compute_and_print_stats(anomaly_dists, "ANOMALOUS distances to center")

    # Optional: quick comparison of means
    mean_norm = normal_dists.mean().item()
    mean_anom = anomaly_dists.mean().item()
    print("\n[SUMMARY] Mean distance comparison:")
    print(f"  mean(normal)   = {mean_norm:.4f}")
    print(f"  mean(anomalous)= {mean_anom:.4f}")
    print(f"  gap (anom - norm) = {mean_anom - mean_norm:.4f}")


if __name__ == "__main__":
    main()

'''

python3 compute_anom_margin.py --cache-path ckpt/floga_teacher_cache_semisupervised.pt --train-csv "../FLOGA/finetuning data/floga_splits/images_train_semisupervised.csv" --data-root ../FLOGA

'''