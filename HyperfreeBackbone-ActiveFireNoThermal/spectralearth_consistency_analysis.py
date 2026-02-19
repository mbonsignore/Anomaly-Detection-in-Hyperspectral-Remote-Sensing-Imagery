#!/usr/bin/env python3
"""
SpectralEarth feature-consistency analysis (HyperFree).

Given 3 folders with matching .pt filenames:
  - hyperspectral
  - multispectral (mean)
  - multispectral (SRF)

Performs:
1) Pairwise distances for matching samples (cosine + L2)
2) Random baseline distances (mismatched pairs)
3) Relative distance ratios R = paired / random (per pair type)
4) Patch-level comparison (mean over patches)
5) (Optional) Linear CKA similarity between representations

Outputs:
- summary.json with key statistics
- distances.csv with per-sample metrics (optionally limited by --max-samples)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def list_pt_files(folder: Path) -> Dict[str, Path]:
    files = {}
    for p in folder.glob("*.pt"):
        files[p.name] = p
    return files


def intersect_filenames(a: Dict[str, Path], b: Dict[str, Path], c: Dict[str, Path]) -> List[str]:
    names = sorted(set(a.keys()) & set(b.keys()) & set(c.keys()))
    return names


def load_feature(path: Path, device: str = "cpu") -> torch.Tensor:
    t = torch.load(path, map_location=device)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{path} does not contain a torch.Tensor.")
    if t.dim() != 3:
        raise ValueError(f"{path} expected [C,H,W], got {tuple(t.shape)}")
    return t


def flatten_global(fm: torch.Tensor) -> torch.Tensor:
    # [C,H,W] -> [C*H*W]
    return fm.reshape(-1)


def flatten_patches(fm: torch.Tensor) -> torch.Tensor:
    # [C,H,W] -> [P,C] where P=H*W
    C, H, W = fm.shape
    return fm.permute(1, 2, 0).contiguous().view(H * W, C)


def cosine_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    # 1 - cos sim
    x = x.float()
    y = y.float()
    xn = x.norm(p=2)
    yn = y.norm(p=2)
    if xn.item() < eps or yn.item() < eps:
        return float("nan")
    cos = torch.dot(x, y) / (xn * yn + eps)
    return (1.0 - cos).item()


def l2_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x.float() - y.float()).norm(p=2).item()


def patchwise_distances(fm_a: torch.Tensor, fm_b: torch.Tensor) -> Tuple[float, float]:
    """
    Compare patch embeddings position-wise.
    Returns (mean cosine distance over patches, mean L2 distance over patches).
    """
    pa = flatten_patches(fm_a)  # [P,C]
    pb = flatten_patches(fm_b)  # [P,C]
    if pa.shape != pb.shape:
        raise ValueError(f"Patch shapes mismatch: {pa.shape} vs {pb.shape}")
    # cosine per patch
    eps = 1e-12
    na = pa.norm(dim=1)  # [P]
    nb = pb.norm(dim=1)
    denom = (na * nb + eps)
    cos = (pa * pb).sum(dim=1) / denom
    cos_dist = (1.0 - cos).mean().item()
    l2 = (pa - pb).norm(dim=1).mean().item()
    return cos_dist, l2


# ----------------------------
# Linear CKA (optional)
# ----------------------------

def center_gram(K: torch.Tensor) -> torch.Tensor:
    # K: [n,n]
    n = K.size(0)
    if n < 2:
        return K
    one_n = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Linear CKA using centered Gram matrices.
    X: [n,d], Y: [n,d2]
    """
    X = X.float()
    Y = Y.float()
    K = X @ X.t()
    L = Y @ Y.t()
    Kc = center_gram(K)
    Lc = center_gram(L)
    hsic = (Kc * Lc).sum()
    norm_x = torch.sqrt((Kc * Kc).sum() + eps)
    norm_y = torch.sqrt((Lc * Lc).sum() + eps)
    return (hsic / (norm_x * norm_y + eps)).item()


# ----------------------------
# Main analysis
# ----------------------------

@dataclass
class PairMetrics:
    name: str
    # global flatten distances
    cos_H_M1: float
    cos_H_M2: float
    cos_M1_M2: float
    l2_H_M1: float
    l2_H_M2: float
    l2_M1_M2: float
    # patchwise distances
    pcos_H_M1: float
    pcos_H_M2: float
    pcos_M1_M2: float
    pl2_H_M1: float
    pl2_H_M2: float
    pl2_M1_M2: float


def summarize(values: List[float]) -> Dict[str, float]:
    vals = [v for v in values if v == v and math.isfinite(v)]  # drop NaN/inf
    if not vals:
        return {"count": 0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def q(p: float) -> float:
        if n == 1:
            return vals_sorted[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals_sorted[lo]
        w = idx - lo
        return (1 - w) * vals_sorted[lo] + w * vals_sorted[hi]

    mean = sum(vals_sorted) / n
    return {
        "count": n,
        "mean": mean,
        "std": (sum((x - mean) ** 2 for x in vals_sorted) / max(1, n - 1)) ** 0.5,
        "min": vals_sorted[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "max": vals_sorted[-1],
    }


def _finite(vals):
    return np.array([v for v in vals if (v == v) and math.isfinite(v)], dtype=np.float64)

def plot_hist_overlay(a, b, title, xlabel, out_path, bins=80, logx=False):
    a = _finite(a); b = _finite(b)
    if a.size == 0 or b.size == 0:
        return
    plt.figure()
    if logx:
        a2 = a[a > 0]; b2 = b[b > 0]
        if a2.size == 0 or b2.size == 0:
            return
        lo = min(a2.min(), b2.min())
        hi = max(a2.max(), b2.max())
        edges = np.logspace(np.log10(lo), np.log10(hi), bins)
        plt.hist(a2, bins=edges, density=True, alpha=0.5, label="paired")
        plt.hist(b2, bins=edges, density=True, alpha=0.5, label="random")
        plt.xscale("log")
    else:
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        edges = np.linspace(lo, hi, bins)
        plt.hist(a, bins=edges, density=True, alpha=0.5, label="paired")
        plt.hist(b, bins=edges, density=True, alpha=0.5, label="random")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_ecdf(a, b, title, xlabel, out_path, logx=False):
    a = np.sort(_finite(a)); b = np.sort(_finite(b))
    if a.size == 0 or b.size == 0:
        return
    ya = np.arange(1, a.size + 1) / a.size
    yb = np.arange(1, b.size + 1) / b.size
    plt.figure()
    plt.plot(a, ya, label="paired")
    plt.plot(b, yb, label="random")
    if logx:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_box_three(data_triplet, labels, title, ylabel, out_path, logy=False):
    # data_triplet: [list1, list2, list3]
    d = [_finite(x) for x in data_triplet]
    plt.figure()
    plt.boxplot(d, labels=labels, showfliers=False)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_scatter(x, y, title, xlabel, ylabel, out_path, sample=20000):
    x = _finite(x); y = _finite(y)
    if x.size == 0 or y.size == 0:
        return
    n = min(x.size, y.size)
    if n > sample:
        idx = np.random.choice(n, size=sample, replace=False)
        x = x[idx]; y = y[idx]
    plt.figure()
    plt.scatter(x, y, s=2, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def auc_prob_paired_smaller_than_random(paired, random_vals):
    """
    Returns P(paired < random), an AUC-like separability measure in [0,1].
    0.5 ~ no separation, 1.0 ~ perfect (paired always smaller).
    Uses rank-based Mann–Whitney U equivalence (O((n+m)log(n+m))).
    """
    a = _finite(paired)
    b = _finite(random_vals)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # concatenate and rank
    x = np.concatenate([a, b])
    # argsort twice gives ranks 0..N-1
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    ra = ranks[: a.size]
    # Mann–Whitney U for "a < b" can be derived from rank sums
    U = ra.sum() - (a.size * (a.size - 1) / 2.0)
    # probability that a < b is U / (n*m)
    return float(U / (a.size * b.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyperspectral", type=str, required=True, help="Folder with hyperspectral features (.pt)")
    ap.add_argument("--ms-mean", type=str, required=True, help="Folder with multispectral arithmetic mean features (.pt)")
    ap.add_argument("--ms-srf", type=str, required=True, help="Folder with multispectral SRF-based features (.pt)")
    ap.add_argument("--out-dir", type=str, default="analysis_out", help="Output directory")
    ap.add_argument("--max-samples", type=int, default=0, help="If >0, analyze only this many matched samples")
    ap.add_argument("--random-baseline", type=int, default=20000, help="Number of random mismatched pairs for baseline")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0 (loading features)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compute-cka", action="store_true", help="Compute linear CKA on sampled embeddings")
    ap.add_argument("--cka-samples", type=int, default=5000, help="How many samples for CKA (uses flattened global vectors)")
    ap.add_argument("--make-plots", action="store_true", help="Generate plots into out-dir/plots")
    ap.add_argument("--plot-bins", type=int, default=80)
    args = ap.parse_args()

    set_seed(args.seed)

    H_dir = Path(args.hyperspectral)
    M1_dir = Path(args.ms_mean)
    M2_dir = Path(args.ms_srf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H_files = list_pt_files(H_dir)
    M1_files = list_pt_files(M1_dir)
    M2_files = list_pt_files(M2_dir)

    names = intersect_filenames(H_files, M1_files, M2_files)
    if not names:
        raise SystemExit("No common .pt filenames across the three folders.")

    if args.max_samples and args.max_samples > 0:
        names = names[: args.max_samples]

    # Basic shape sanity check on first sample
    first = names[0]
    fmH = load_feature(H_files[first], device=args.device)
    fmM1 = load_feature(M1_files[first], device=args.device)
    fmM2 = load_feature(M2_files[first], device=args.device)
    shapes = {
        "H": tuple(fmH.shape),
        "M1": tuple(fmM1.shape),
        "M2": tuple(fmM2.shape),
    }
    if shapes["H"] != shapes["M1"] or shapes["H"] != shapes["M2"]:
        print("[WARN] Feature map shapes differ across modalities for the first sample:")
        print(shapes)

    metrics: List[PairMetrics] = []

    # For CKA we’ll optionally collect flattened vectors for a subset
    cka_collect = args.compute_cka
    cka_n = min(args.cka_samples, len(names)) if cka_collect else 0
    cka_indices = set(random.sample(range(len(names)), k=cka_n)) if cka_collect and cka_n > 0 else set()
    XH = []
    XM1 = []
    XM2 = []

    for idx, name in enumerate(tqdm(names, desc="Paired distances")):
        fH = load_feature(H_files[name], device=args.device)
        fM1 = load_feature(M1_files[name], device=args.device)
        fM2 = load_feature(M2_files[name], device=args.device)

        gH = flatten_global(fH)
        gM1 = flatten_global(fM1)
        gM2 = flatten_global(fM2)

        cos_H_M1 = cosine_distance(gH, gM1)
        cos_H_M2 = cosine_distance(gH, gM2)
        cos_M1_M2 = cosine_distance(gM1, gM2)
        l2_H_M1 = l2_distance(gH, gM1)
        l2_H_M2 = l2_distance(gH, gM2)
        l2_M1_M2 = l2_distance(gM1, gM2)

        pcos_H_M1, pl2_H_M1 = patchwise_distances(fH, fM1)
        pcos_H_M2, pl2_H_M2 = patchwise_distances(fH, fM2)
        pcos_M1_M2, pl2_M1_M2 = patchwise_distances(fM1, fM2)

        metrics.append(
            PairMetrics(
                name=name,
                cos_H_M1=cos_H_M1,
                cos_H_M2=cos_H_M2,
                cos_M1_M2=cos_M1_M2,
                l2_H_M1=l2_H_M1,
                l2_H_M2=l2_H_M2,
                l2_M1_M2=l2_M1_M2,
                pcos_H_M1=pcos_H_M1,
                pcos_H_M2=pcos_H_M2,
                pcos_M1_M2=pcos_M1_M2,
                pl2_H_M1=pl2_H_M1,
                pl2_H_M2=pl2_H_M2,
                pl2_M1_M2=pl2_M1_M2,
            )
        )

        if cka_collect and idx in cka_indices:
            # store CPU float32 for stability
            XH.append(gH.detach().to("cpu", dtype=torch.float32))
            XM1.append(gM1.detach().to("cpu", dtype=torch.float32))
            XM2.append(gM2.detach().to("cpu", dtype=torch.float32))

    # ----------------------------
    # Random baseline (mismatched pairs)
    # ----------------------------
    # We compute random distances within the SAME modality to represent "different image" distances.
    # And also (optionally) cross-modality random mismatch baselines.
    K = min(args.random_baseline, len(names) * 3)  # avoid silly huge when small dataset
    idxs = list(range(len(names)))

    def random_pairs(K: int) -> List[Tuple[int, int]]:
        pairs = []
        for _ in range(K):
            i = random.choice(idxs)
            j = random.choice(idxs)
            while j == i:
                j = random.choice(idxs)
            pairs.append((i, j))
        return pairs

    base_pairs = random_pairs(K)

    # Cache: to avoid loading twice too much, we’ll just load on demand (still OK for baseline size ~20k).
    # If IO is slow, reduce --random-baseline.
    rand_cos_HH, rand_l2_HH = [], []
    rand_cos_M1M1, rand_l2_M1M1 = [], []
    rand_cos_M2M2, rand_l2_M2M2 = [], []

    # cross-modality mismatched
    rand_cos_HM1, rand_l2_HM1 = [], []
    rand_cos_HM2, rand_l2_HM2 = [], []
    rand_cos_M1M2, rand_l2_M1M2 = [], []

    for i, j in tqdm(base_pairs, desc="Random baselines"):
        ni = names[i]
        nj = names[j]

        fHi = load_feature(H_files[ni], device=args.device)
        fHj = load_feature(H_files[nj], device=args.device)
        gi = flatten_global(fHi); gj = flatten_global(fHj)
        rand_cos_HH.append(cosine_distance(gi, gj))
        rand_l2_HH.append(l2_distance(gi, gj))

        fM1i = load_feature(M1_files[ni], device=args.device)
        fM1j = load_feature(M1_files[nj], device=args.device)
        gi = flatten_global(fM1i); gj = flatten_global(fM1j)
        rand_cos_M1M1.append(cosine_distance(gi, gj))
        rand_l2_M1M1.append(l2_distance(gi, gj))

        fM2i = load_feature(M2_files[ni], device=args.device)
        fM2j = load_feature(M2_files[nj], device=args.device)
        gi = flatten_global(fM2i); gj = flatten_global(fM2j)
        rand_cos_M2M2.append(cosine_distance(gi, gj))
        rand_l2_M2M2.append(l2_distance(gi, gj))

        # cross-modality mismatched (H_i vs M1_j etc.)
        gi = flatten_global(fHi); gj = flatten_global(fM1j)
        rand_cos_HM1.append(cosine_distance(gi, gj))
        rand_l2_HM1.append(l2_distance(gi, gj))

        gi = flatten_global(fHi); gj = flatten_global(fM2j)
        rand_cos_HM2.append(cosine_distance(gi, gj))
        rand_l2_HM2.append(l2_distance(gi, gj))

        gi = flatten_global(fM1i); gj = flatten_global(fM2j)
        rand_cos_M1M2.append(cosine_distance(gi, gj))
        rand_l2_M1M2.append(l2_distance(gi, gj))

    # ----------------------------
    # Relative distance ratios (paired / random)
    # ----------------------------
    paired_cos_HM1 = [m.cos_H_M1 for m in metrics]
    paired_cos_HM2 = [m.cos_H_M2 for m in metrics]
    paired_cos_M1M2 = [m.cos_M1_M2 for m in metrics]

    paired_l2_HM1 = [m.l2_H_M1 for m in metrics]
    paired_l2_HM2 = [m.l2_H_M2 for m in metrics]
    paired_l2_M1M2 = [m.l2_M1_M2 for m in metrics]

    # Use same-modality random baselines for “different images” and cross-modality mismatched as extra reference.
    # Ratio is informative if baselines are not near-zero.
    def safe_ratio(a_list: List[float], b_list: List[float]) -> Dict[str, float]:
        a = [x for x in a_list if x == x and math.isfinite(x)]
        b = [x for x in b_list if x == x and math.isfinite(x)]
        if not a or not b:
            return {"count": 0}
        b_mean = sum(b) / len(b)
        if abs(b_mean) < 1e-12:
            return {"count": len(a), "mean": float("inf")}
        ratios = [x / b_mean for x in a]
        return summarize(ratios)

    # ----------------------------
    # CKA
    # ----------------------------
    cka_results = {}
    if cka_collect and XH:
        XH_t = torch.stack(XH, dim=0)   # [n,d]
        XM1_t = torch.stack(XM1, dim=0)
        XM2_t = torch.stack(XM2, dim=0)
        # (Optional) normalize vectors to reduce magnitude effects
        # Uncomment if you prefer direction-only similarity:
        # XH_t = torch.nn.functional.normalize(XH_t, dim=1)
        # XM1_t = torch.nn.functional.normalize(XM1_t, dim=1)
        # XM2_t = torch.nn.functional.normalize(XM2_t, dim=1)

        cka_results = {
            "n": XH_t.size(0),
            "linear_cka_H_M1": linear_cka(XH_t, XM1_t),
            "linear_cka_H_M2": linear_cka(XH_t, XM2_t),
            "linear_cka_M1_M2": linear_cka(XM1_t, XM2_t),
        }

    # ----------------------------
    # Save per-sample CSV
    # ----------------------------
    csv_path = out_dir / "distances.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name",
            "cos_H_M1","cos_H_M2","cos_M1_M2",
            "l2_H_M1","l2_H_M2","l2_M1_M2",
            "pcos_H_M1","pcos_H_M2","pcos_M1_M2",
            "pl2_H_M1","pl2_H_M2","pl2_M1_M2",
        ])
        for m in metrics:
            w.writerow([
                m.name,
                m.cos_H_M1, m.cos_H_M2, m.cos_M1_M2,
                m.l2_H_M1, m.l2_H_M2, m.l2_M1_M2,
                m.pcos_H_M1, m.pcos_H_M2, m.pcos_M1_M2,
                m.pl2_H_M1, m.pl2_H_M2, m.pl2_M1_M2,
            ])

    # ----------------------------
    # Summary JSON
    # ----------------------------
    summary = {
        "folders": {
            "hyperspectral": str(H_dir),
            "ms_mean": str(M1_dir),
            "ms_srf": str(M2_dir),
        },
        "num_matched_samples": len(names),
        "first_sample_shapes": shapes,
        "paired_global_cosine": {
            "H_M1": summarize(paired_cos_HM1),
            "H_M2": summarize(paired_cos_HM2),
            "M1_M2": summarize(paired_cos_M1M2),
        },
        "paired_global_l2": {
            "H_M1": summarize(paired_l2_HM1),
            "H_M2": summarize(paired_l2_HM2),
            "M1_M2": summarize(paired_l2_M1M2),
        },
        "random_same_modality_global_cosine": {
            "H_H": summarize(rand_cos_HH),
            "M1_M1": summarize(rand_cos_M1M1),
            "M2_M2": summarize(rand_cos_M2M2),
        },
        "random_same_modality_global_l2": {
            "H_H": summarize(rand_l2_HH),
            "M1_M1": summarize(rand_l2_M1M1),
            "M2_M2": summarize(rand_l2_M2M2),
        },
        "random_cross_modality_global_cosine": {
            "H_M1": summarize(rand_cos_HM1),
            "H_M2": summarize(rand_cos_HM2),
            "M1_M2": summarize(rand_cos_M1M2),
        },
        "random_cross_modality_global_l2": {
            "H_M1": summarize(rand_l2_HM1),
            "H_M2": summarize(rand_l2_HM2),
            "M1_M2": summarize(rand_l2_M1M2),
        },
        "relative_ratio_paired_over_random_same_modality": {
            # paired cross-modality vs random same-modality "different images"
            # (cosine)
            "cos_H_M1_over_H_H": safe_ratio(paired_cos_HM1, rand_cos_HH),
            "cos_H_M2_over_H_H": safe_ratio(paired_cos_HM2, rand_cos_HH),
            "cos_M1_M2_over_M1_M1": safe_ratio(paired_cos_M1M2, rand_cos_M1M1),
            # (l2)
            "l2_H_M1_over_H_H": safe_ratio(paired_l2_HM1, rand_l2_HH),
            "l2_H_M2_over_H_H": safe_ratio(paired_l2_HM2, rand_l2_HH),
            "l2_M1_M2_over_M1_M1": safe_ratio(paired_l2_M1M2, rand_l2_M1M1),
        },
        "cka": cka_results,
        "notes": [
            "For invariance you want paired distances << random distances, hence ratios << 1.",
            "Cosine is often more stable than L2 when spectral compression changes feature magnitudes.",
            "Patchwise metrics (pcos/pl2) compare patches position-wise; global metrics flatten everything.",
        ],
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # ----------------------------
    # Extra thesis-friendly numbers + plots
    # ----------------------------
    # AUC-style separation: P(paired < random)
    separation = {
        "cos_H_M1_vs_H_H": auc_prob_paired_smaller_than_random(paired_cos_HM1, rand_cos_HH),
        "cos_H_M2_vs_H_H": auc_prob_paired_smaller_than_random(paired_cos_HM2, rand_cos_HH),
        "cos_M1_M2_vs_M1_M1": auc_prob_paired_smaller_than_random(paired_cos_M1M2, rand_cos_M1M1),
        "l2_H_M1_vs_H_H": auc_prob_paired_smaller_than_random(paired_l2_HM1, rand_l2_HH),
        "l2_H_M2_vs_H_H": auc_prob_paired_smaller_than_random(paired_l2_HM2, rand_l2_HH),
        "l2_M1_M2_vs_M1_M1": auc_prob_paired_smaller_than_random(paired_l2_M1M2, rand_l2_M1M1),
    }
    summary["separation_auc_like"] = separation
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if args.make_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- Cosine (linear scale is fine) ---
        plot_hist_overlay(paired_cos_HM1, rand_cos_HH,
                          "Cosine distance: paired H-M1 vs random H-H",
                          "cosine distance (1 - cos)", plots_dir / "hist_cos_HM1_vs_HH.png",
                          bins=args.plot_bins, logx=False)
        plot_hist_overlay(paired_cos_HM2, rand_cos_HH,
                          "Cosine distance: paired H-M2 vs random H-H",
                          "cosine distance (1 - cos)", plots_dir / "hist_cos_HM2_vs_HH.png",
                          bins=args.plot_bins, logx=False)
        plot_hist_overlay(paired_cos_M1M2, rand_cos_M1M1,
                          "Cosine distance: paired M1-M2 vs random M1-M1",
                          "cosine distance (1 - cos)", plots_dir / "hist_cos_M1M2_vs_M1M1.png",
                          bins=args.plot_bins, logx=False)

        plot_ecdf(paired_cos_HM1, rand_cos_HH,
                  "ECDF cosine: paired H-M1 vs random H-H",
                  "cosine distance (1 - cos)", plots_dir / "ecdf_cos_HM1_vs_HH.png")
        plot_ecdf(paired_cos_HM2, rand_cos_HH,
                  "ECDF cosine: paired H-M2 vs random H-H",
                  "cosine distance (1 - cos)", plots_dir / "ecdf_cos_HM2_vs_HH.png")
        plot_ecdf(paired_cos_M1M2, rand_cos_M1M1,
                  "ECDF cosine: paired M1-M2 vs random M1-M1",
                  "cosine distance (1 - cos)", plots_dir / "ecdf_cos_M1M2_vs_M1M1.png")

        # --- L2 (often better on log scale) ---
        plot_hist_overlay(paired_l2_HM1, rand_l2_HH,
                          "L2 distance: paired H-M1 vs random H-H",
                          "L2 distance", plots_dir / "hist_l2_HM1_vs_HH_log.png",
                          bins=args.plot_bins, logx=True)
        plot_hist_overlay(paired_l2_HM2, rand_l2_HH,
                          "L2 distance: paired H-M2 vs random H-H",
                          "L2 distance", plots_dir / "hist_l2_HM2_vs_HH_log.png",
                          bins=args.plot_bins, logx=True)
        plot_hist_overlay(paired_l2_M1M2, rand_l2_M1M1,
                          "L2 distance: paired M1-M2 vs random M1-M1",
                          "L2 distance", plots_dir / "hist_l2_M1M2_vs_M1M1_log.png",
                          bins=args.plot_bins, logx=True)

        plot_ecdf(paired_l2_HM1, rand_l2_HH,
                  "ECDF L2: paired H-M1 vs random H-H",
                  "L2 distance", plots_dir / "ecdf_l2_HM1_vs_HH_log.png", logx=True)
        plot_ecdf(paired_l2_HM2, rand_l2_HH,
                  "ECDF L2: paired H-M2 vs random H-H",
                  "L2 distance", plots_dir / "ecdf_l2_HM2_vs_HH_log.png", logx=True)
        plot_ecdf(paired_l2_M1M2, rand_l2_M1M1,
                  "ECDF L2: paired M1-M2 vs random M1-M1",
                  "L2 distance", plots_dir / "ecdf_l2_M1M2_vs_M1M1_log.png", logx=True)

        # --- Boxplots across the 3 paired comparisons ---
        plot_box_three([paired_cos_HM1, paired_cos_HM2, paired_cos_M1M2],
                       ["H-M1", "H-M2", "M1-M2"],
                       "Paired cosine distances across modality pairs",
                       "cosine distance (1 - cos)", plots_dir / "box_paired_cos.png")
        plot_box_three([paired_l2_HM1, paired_l2_HM2, paired_l2_M1M2],
                       ["H-M1", "H-M2", "M1-M2"],
                       "Paired L2 distances across modality pairs (log y)",
                       "L2 distance", plots_dir / "box_paired_l2_log.png", logy=True)

        # --- Scatter: which MS is closer to H? ---
        plot_scatter(paired_cos_HM1, paired_cos_HM2,
                     "Per-sample cosine: H-M1 vs H-M2 (lower is better)",
                     "cos(H,M1)", "cos(H,M2)", plots_dir / "scatter_cos_HM1_vs_HM2.png")
        plot_scatter(paired_l2_HM1, paired_l2_HM2,
                     "Per-sample L2: H-M1 vs H-M2 (lower is better)",
                     "l2(H,M1)", "l2(H,M2)", plots_dir / "scatter_l2_HM1_vs_HM2.png")

        print(f"[OK] Plots saved to: {plots_dir}")
        print("[INFO] AUC-like separation (P(paired < random)):", separation)

    print(f"[OK] Saved per-sample distances to: {csv_path}")
    print(f"[OK] Saved summary to: {summary_path}")

    # quick console recap
    def shortline(title: str, stats: Dict[str, float]) -> None:
        if "count" in stats and stats["count"] == 0:
            print(f"{title}: (no data)")
            return
        print(f"{title}: mean={stats.get('mean'):.6g}  median={stats.get('median'):.6g}  p75={stats.get('p75'):.6g}")

    print("\n=== Paired cosine distances (lower is better) ===")
    shortline("H-M1", summary["paired_global_cosine"]["H_M1"])
    shortline("H-M2", summary["paired_global_cosine"]["H_M2"])
    shortline("M1-M2", summary["paired_global_cosine"]["M1_M2"])

    print("\n=== Random same-modality cosine distances (reference) ===")
    shortline("H-H", summary["random_same_modality_global_cosine"]["H_H"])
    shortline("M1-M1", summary["random_same_modality_global_cosine"]["M1_M1"])
    shortline("M2-M2", summary["random_same_modality_global_cosine"]["M2_M2"])

    print("\n=== Ratio paired / random (<< 1 means good invariance) ===")
    shortline("cos(H-M1)/cos(H-H)", summary["relative_ratio_paired_over_random_same_modality"]["cos_H_M1_over_H_H"])
    shortline("cos(H-M2)/cos(H-H)", summary["relative_ratio_paired_over_random_same_modality"]["cos_H_M2_over_H_H"])
    shortline("cos(M1-M2)/cos(M1-M1)", summary["relative_ratio_paired_over_random_same_modality"]["cos_M1_M2_over_M1_M1"])

    if cka_results:
        print("\n=== Linear CKA (higher is more similar; ~1 is very similar) ===")
        print(f"n={cka_results['n']}, H-M1={cka_results['linear_cka_H_M1']:.6f}, H-M2={cka_results['linear_cka_H_M2']:.6f}, M1-M2={cka_results['linear_cka_M1_M2']:.6f}")


if __name__ == "__main__":
    main()


'''# Example usage:
python3 spectralearth_consistency_analysis.py \
--hyperspectral ../../../data/datasets/hyperfree/Extracted\ Features/features\ spectralearth \
--ms-mean ../../../data/datasets/hyperfree/Extracted\ Features/features\ spectralearth\ 7\ bands \
--ms-srf ../../../data/datasets/hyperfree/Extracted\ Features/features\ spectralearth\ 7\ bands\ srf \
--out-dir consistency_analysis_spectralearth \
--random-baseline 20000 \
--compute-cka \
--cka-samples 5000
--make-plots
'''