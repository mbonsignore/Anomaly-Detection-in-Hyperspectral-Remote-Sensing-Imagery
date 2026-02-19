#!/usr/bin/env python3
"""
Analyze teacher cache distances to choose a good margin m
for the semi-supervised PatchCore-friendly finetuning.

CHANGELOG vs previous version:
- Now matches cache entries by *filename stem* (e.g. 'ID_pre'),
  not by full path, to avoid mismatches when data_root or
  absolute paths differ between runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import csv


def load_teacher_cache(cache_path: Path) -> Dict[str, torch.Tensor]:
    if not cache_path.is_file():
        raise SystemExit(f"Teacher cache not found: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    cache_raw: Dict[str, torch.Tensor] = payload["cache"]

    # Build mapping: stem -> embedding
    cache_by_stem: Dict[str, torch.Tensor] = {}
    for path_str, emb in cache_raw.items():
        stem = Path(path_str).stem  # e.g. "FLOGA_001_pre"
        if stem in cache_by_stem:
            # In principle this shouldn't happen with your setup,
            # but we guard against accidental duplicates.
            print(f"[WARN] Duplicate stem in cache: {stem} (keeping first)")
            continue
        cache_by_stem[stem] = emb

    print(f"[INFO] Raw cache size: {len(cache_raw)} embeddings")
    print(f"[INFO] Unique stems in cache: {len(cache_by_stem)}")
    if len(cache_by_stem) > 0:
        any_emb = next(iter(cache_by_stem.values()))
        print(f"[INFO] Embedding dim: {any_emb.numel()}")

    return cache_by_stem


def load_labels_from_csvs(
    csv_paths: List[Path],
    cache_by_stem: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From the semisupervised CSVs, extract:
      - labels: 0 normal, 1 anomalous
      - distances: ||z - c|| will be computed later; for now
        we just collect stems and labels.

    Matching is done on sample_name == stem used in cache_by_stem.
    """
    # We'll collect (stem, label) only for items present in cache.
    stems: List[str] = []
    labels: List[int] = []

    print("[INFO] Loading labels from CSVs:")
    for csv_path in csv_paths:
        print(f"  - {csv_path}")
        if not csv_path.is_file():
            raise SystemExit(f"CSV not found: {csv_path}")

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            expected = {"sample_name", "folder", "label"}
            if not expected.issubset(reader.fieldnames or []):
                raise SystemExit(
                    f"CSV {csv_path} must contain columns {sorted(expected)}, "
                    f"found {reader.fieldnames}"
                )

            for row in reader:
                stem = row["sample_name"]  # e.g. "FLOGA_001_pre"
                label = int(row["label"])
                if stem not in cache_by_stem:
                    continue
                stems.append(stem)
                labels.append(label)

    labels_np = np.asarray(labels, dtype=np.int64)
    print(f"[INFO] Matched samples: {len(stems)}")
    print(f"[WARN] {len(cache_by_stem) - len(stems)} cached embeddings had no label")
    return np.asarray(stems, dtype=object), labels_np


def compute_distances_to_center(
    stems: np.ndarray,
    labels: np.ndarray,
    cache_by_stem: Dict[str, torch.Tensor],
    center: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Compute ||z - c||_2 for each stem. If center is None, compute it as
    the mean of *normal* embeddings in the matched set.
    """
    # Stack embeddings for matched stems
    embs = torch.stack([cache_by_stem[str(stem)] for stem in stems], dim=0)  # [N,D]

    if center is None:
        # Center from normal samples only
        normal_mask = labels == 0
        if normal_mask.sum() == 0:
            raise RuntimeError("No normal samples found to estimate center.")
        center = embs[normal_mask].mean(dim=0, keepdim=False)  # [D]

    center = center.view(1, -1)  # [1,D]
    diff = embs - center
    dists = torch.norm(diff, dim=1)  # [N]
    return dists.numpy(), center.squeeze(0)


def summarize_distances(dists: np.ndarray, labels: np.ndarray) -> None:
    """
    Print statistics for normals vs anomalies separately and suggest a margin.
    """
    normal_mask = labels == 0
    anom_mask = labels == 1

    print("\n========== DISTANCE STATISTICS ==========")

    def stats(mask: np.ndarray, name: str) -> None:
        subset = dists[mask]
        if subset.size == 0:
            print(f"[WARN] No samples for {name}")
            return
        print(f"\n[{name}] n={subset.size}")
        print(f"  min   = {subset.min():.6f}")
        print(f"  max   = {subset.max():.6f}")
        print(f"  mean  = {subset.mean():.6f}")
        print(f"  std   = {subset.std():.6f}")
        for p in [50, 75, 90, 95, 97, 99]:
            val = np.percentile(subset, p)
            print(f"  p{p:02d}  = {val:.6f}")

    stats(normal_mask, "NORMAL")
    stats(anom_mask, "ANOMALOUS")

    if normal_mask.sum() > 0:
        normal_95 = np.percentile(dists[normal_mask], 95)
        print(f"\n[Suggestion] A reasonable starting margin m could be ≈ p95(normal) = {normal_95:.6f}")
        print("You can try values around this (e.g. 0.8×, 1.0×, 1.2×) in the semi-supervised loss.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analyze teacher-cache distances (by stem) to choose margin m."
    )
    ap.add_argument(
        "--teacher-cache",
        type=str,
        required=True,
        help="Path to floga_teacher_cache.pt (with {'cache': dict(path->embedding)}).",
    )
    ap.add_argument(
        "--csv",
        type=str,
        nargs="+",
        required=True,
        help="One or more semisupervised image CSVs (train/val) with columns sample_name,folder,label.",
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Unused now (kept for compatibility).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = Path(args.teacher_cache)
    csv_paths = [Path(p) for p in args.csv]

    cache_by_stem = load_teacher_cache(cache_path)
    stems, labels = load_labels_from_csvs(csv_paths, cache_by_stem)

    if len(stems) == 0:
        print("[ERROR] No overlap between cache stems and CSV sample_name.")
        print("        Check that you are using the cache computed with the same FLOGA split.")
        return

    dists, center = compute_distances_to_center(stems, labels, cache_by_stem, center=None)
    summarize_distances(dists, labels)


if __name__ == "__main__":
    main()
