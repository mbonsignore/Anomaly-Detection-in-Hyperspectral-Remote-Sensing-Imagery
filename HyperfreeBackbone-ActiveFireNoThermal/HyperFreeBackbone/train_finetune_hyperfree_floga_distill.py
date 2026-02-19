#!/usr/bin/env python3
"""Fine-tune HyperFree on FLOGA with center + distillation losses,
with optional caching of teacher embeddings + center, and PatchCore-based
evaluation on the validation set.

PatchCore config (fixed):
    - sampling_ratio = 0.1
    - coreset_method = "random"
    - alternating split on NORMAL val samples: half for memory bank,
      half for normal test, anomalies always in test.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode, resize

from config import BackboneConfig
from model import build_image_encoder, load_weights
from utils.common import LayerNorm2d

# ============================
#  IMPORT PATCHCORE MODULES
#  (from ../../patchcore, WITHOUT breaking `utils.common`)
# ============================
from importlib.machinery import SourceFileLoader
import sys

PATCHCORE_DIR = (Path(__file__).resolve().parent / "../../patchcore").resolve()

# 1) Load PatchCore's utils.py as its own module
pc_utils = SourceFileLoader(
    "patchcore_utils", str(PATCHCORE_DIR / "utils.py")
).load_module()

# 2) Tell Python that *for this process* the name "utils" refers to PatchCore's utils
#    (this happens AFTER we already imported from utils.common above, so it does
#    not break LayerNorm2d, config, model, etc.)
sys.modules["patchcore_utils"] = pc_utils
sys.modules["utils"] = pc_utils

# 3) Now load memory_bank.py; its "import utils" will see the PatchCore utils
pc_memory_bank = SourceFileLoader(
    "patchcore_memory_bank", str(PATCHCORE_DIR / "memory_bank.py")
).load_module()

# Fixed PatchCore config you requested
PATCHCORE_SAMPLING_RATIO = 0.1
PATCHCORE_CORESET_METHOD = "random"


class FlogaCsvDataset(Dataset):
    """Load FLOGA .pt tensors listed in a CSV with labels."""

    def __init__(
        self,
        csv_path: Path,
        data_root: Path,
        wavelengths: np.ndarray,
        img_size: int,
        only_label: int | None = None,
    ) -> None:
        self.data_root = data_root
        self.img_size = img_size
        self.wavelengths = torch.as_tensor(wavelengths, dtype=torch.float32)
        self.samples: List[Tuple[str, str, int]] = []

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            expected = {"sample_name", "folder", "label"}
            if not expected.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"CSV {csv_path} must include columns {sorted(expected)}; "
                    f"found {reader.fieldnames}"
                )
            for row in reader:
                label = int(row["label"])
                if only_label is not None and label != only_label:
                    continue
                self.samples.append((row["sample_name"], row["folder"], label))

        if not self.samples:
            raise ValueError(f"No samples found in {csv_path} after filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample_name, folder, label = self.samples[index]
        path = self.data_root / folder / f"{sample_name}.pt"

        try:
            tensor = torch.load(path, map_location="cpu").float()
        except Exception as exc:
            print(f"[WARN] Failed to load {path}: {exc}")
            return self.__getitem__((index + 1) % len(self.samples))

        if tensor.ndim != 3:
            raise ValueError(f"{path} has invalid shape {tuple(tensor.shape)}")

        original_shape = torch.tensor(tensor.shape[-2:], dtype=torch.int)

        tensor = resize(
            tensor,
            size=[self.img_size, self.img_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        # per-channel min-max
        for c in range(tensor.shape[0]):
            band = tensor[c]
            bmin = band.min()
            bmax = band.max()
            if (bmax - bmin) > 0:
                tensor[c] = (band - bmin) / (bmax - bmin)

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "wavelengths": self.wavelengths.clone(),
            "original_shape": original_shape,
            "path": str(path),
        }


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_data_root = base_dir / "../../../../data/datasets/FLOGA"
    default_splits_root = base_dir / "../../../../data/datasets/FLOGA/finetuning data/floga_splits"

    parser = argparse.ArgumentParser(
        description="Fine-tune HyperFree backbone on FLOGA with center + distillation losses (PatchCore eval)."
    )
    parser.add_argument("--data-root", type=str, default=str(default_data_root))
    parser.add_argument("--splits-root", type=str, default=str(default_splits_root))
    parser.add_argument("--train-csv", type=str, default="images_train.csv")
    parser.add_argument("--val-csv", type=str, default="images_val.csv")
    parser.add_argument(
        "--wavelengths-path",
        type=str,
        default=str(base_dir / "../wavelengths_floga.npy"),
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default=str(base_dir / "ckpt/HyperFree-b.pth"),
    )
    parser.add_argument("--out-ckpt-dir", type=str, default=str(base_dir / "ckpt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(2, (os.cpu_count() or 4) // 2),
    )
    parser.add_argument("--lambda-center", type=float, default=1.0)
    parser.add_argument("--lambda-distill", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--auto-batch", action="store_true", default=True)
    parser.add_argument("--no-auto-batch", dest="auto_batch", action="store_false")
    parser.add_argument("--min-batch-size", type=int, default=4)

    # ===== CACHE OPTIONS (unchanged) =====
    parser.add_argument(
        "--teacher-cache-path",
        type=str,
        default=str(base_dir / "ckpt/floga_teacher_cache.pt"),
        help="Path to save/load teacher embeddings + center.",
    )
    parser.add_argument(
        "--load-teacher-cache",
        action="store_true",
        help="If set, try to load teacher cache from --teacher-cache-path.",
    )
    parser.add_argument(
        "--save-teacher-cache",
        action="store_true",
        help="If set, save computed teacher cache to --teacher-cache-path.",
    )

    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preferred: str) -> torch.device:
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        print("[INFO] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred)


def resolve_csv_path(csv_name: str, splits_root: Path) -> Path:
    csv_path = Path(csv_name)
    if csv_path.is_file():
        return csv_path
    return splits_root / csv_name


def get_embedding(features: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
    """Average over spatial tokens of the last feature map [B,C,H,W] → [B,D]."""
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if features.dim() != 4:
        raise ValueError(f"Expected 4D feature map [B,C,H,W], got {features.shape}")
    return features.flatten(2).mean(dim=2)


def compute_gsd(
    original_shapes: torch.Tensor,
    gsd_meters: float,
    target_img_size: int,
) -> List[float]:
    widths = original_shapes[:, 1].float()
    gsd = gsd_meters * (widths / target_img_size)
    return gsd.tolist()


# --------- PRECOMPUTE TEACHER EMBEDDINGS + CENTER (unchanged) ---------
@torch.no_grad()
def precompute_teacher_cache(
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gsd_meters: float,
    img_size: int,
    use_amp: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Run teacher once on the whole TRAIN loader and cache embeddings by path.
    Also compute the global center from these cached embeddings.
    """
    teacher.eval()
    cache: Dict[str, torch.Tensor] = {}
    sum_embed = None
    total = 0

    with tqdm(loader, desc="Precomputing teacher embeddings", unit="batch") as pbar:
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            batch_wavelengths = batch["wavelengths"]
            encoder_wavelengths = batch_wavelengths[0].tolist()
            gsd = compute_gsd(batch["original_shape"], gsd_meters, img_size)
            paths = batch["path"]  # list of str

            with torch.cuda.amp.autocast(enabled=use_amp):
                feats = teacher(
                    images,
                    test_mode=True,
                    input_wavelength=encoder_wavelengths,
                    GSD=gsd,
                )
                z = get_embedding(feats).float().cpu()  # [B,D] on CPU

            # Fill cache & accumulate for center
            for p, emb in zip(paths, z):
                cache[p] = emb.clone()

            batch_sum = z.sum(dim=0)  # [D]
            sum_embed = batch_sum if sum_embed is None else sum_embed + batch_sum
            total += z.shape[0]

    if total == 0:
        raise RuntimeError("Cannot initialize teacher cache: no samples in loader.")

    center = sum_embed / float(total)  # [D] on CPU
    return cache, center


def freeze_backbone(
    encoder: torch.nn.Module,
    unfreeze_neck_norms: bool = True,
) -> None:
    for param in encoder.parameters():
        param.requires_grad = False

    num_blocks = len(encoder.blocks)
    if num_blocks >= 12:
        unfreeze_blocks = 4
    else:
        unfreeze_blocks = math.ceil(num_blocks / 4)

    for blk in encoder.blocks[-unfreeze_blocks:]:
        for param in blk.parameters():
            param.requires_grad = True

    if unfreeze_neck_norms:
        for module in encoder.neck.modules():
            if isinstance(module, (torch.nn.LayerNorm, LayerNorm2d)):
                for param in module.parameters():
                    param.requires_grad = True


def train_one_epoch(
    teacher: torch.nn.Module,  # unused, kept for signature compatibility
    student: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    center: torch.Tensor,
    gsd_meters: float,
    img_size: int,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    lambda_center: float,
    lambda_distill: float,
    teacher_cache: Dict[str, torch.Tensor],
) -> float:
    """Train student using cached teacher embeddings + fixed center."""
    student.train()
    running_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Training", unit="batch", leave=False):
        labels = batch["label"]
        if labels.max().item() != 0:
            raise ValueError("Training batch contains anomalous samples (label=1).")

        images = batch["image"].to(device, non_blocking=True)
        batch_wavelengths = batch["wavelengths"]
        encoder_wavelengths = batch_wavelengths[0].tolist()
        gsd = compute_gsd(batch["original_shape"], gsd_meters, img_size)
        paths = batch["path"]  # list[str]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # Teacher embeddings from cache (CPU → device)
            z_teacher = torch.stack(
                [teacher_cache[p] for p in paths]
            ).to(device)  # [B,D]

            # Student forward
            student_features = student(
                images,
                test_mode=True,
                input_wavelength=encoder_wavelengths,
                GSD=gsd,
            )
            z_student = get_embedding(student_features)  # [B,D]

            # Center loss (student vs fixed center)
            diff_center = z_student - center.to(z_student.dtype)
            loss_center = diff_center.pow(2).sum(dim=1).mean()

            # Distillation loss (student vs teacher cache)
            diff_distill = z_student - z_teacher.to(z_student.dtype)
            loss_distill = diff_distill.pow(2).sum(dim=1).mean()

            loss = lambda_center * loss_center + lambda_distill * loss_distill

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


# ========= METRIC HELPERS (reused by PatchCore eval) =========
def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos_mask = labels == 1
    n_pos = int(pos_mask.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    ranks = np.empty_like(scores_sorted, dtype=float)
    i = 0
    rank = 1
    while i < len(scores_sorted):
        j = i
        while j < len(scores_sorted) and scores_sorted[j] == scores_sorted[i]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        ranks[i:j] = avg_rank
        rank += (j - i)
        i = j

    sum_pos = ranks[labels_sorted == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order].astype(np.int64)

    tp = 0
    fp = 0
    fn = int(labels_sorted.sum())
    best_f1 = -1.0
    best_thresh = float("inf")

    i = 0
    while i < len(scores_sorted):
        score = scores_sorted[i]
        j = i
        while j < len(scores_sorted) and scores_sorted[j] == score:
            j += 1

        while i < j:
            if labels_sorted[i] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
            i += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = score

    return best_thresh, best_f1


# ========= PATCHCORE EVALUATION (VAL SET) =========
@torch.no_grad()
def evaluate_patchcore(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gsd_meters: float,
    img_size: int,
    use_amp: bool,
) -> Dict[str, float | List[List[int]]]:
    """
    PatchCore-style evaluation on the VALIDATION set:

      - Run encoder on all val samples → feature maps
      - Split NORMAL samples into:
            * half for memory bank (alternating split)
            * half for normal test
      - All ANOMALOUS samples go to anomalous test
      - Build MemoryBank(sampling_ratio=0.1, coreset='random')
      - Compute image-level scores on normal_test + anomalous
      - Return metrics (AUROC, best F1, etc.)
    """

    encoder.eval()

    normal_sets: List[pc_utils.FeatureSet] = []
    anomalous_sets: List[pc_utils.FeatureSet] = []
    label_map: Dict[str, int] = {}

    with tqdm(loader, desc="PatchCore: extracting features (val)", unit="batch", leave=False) as pbar:
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy()  # [B]
            paths = batch["path"]  # list[str]
            batch_wavelengths = batch["wavelengths"]
            encoder_wavelengths = batch_wavelengths[0].tolist()
            gsd = compute_gsd(batch["original_shape"], gsd_meters, img_size)

            with torch.cuda.amp.autocast(enabled=use_amp):
                feats = encoder(
                    images,
                    test_mode=True,
                    input_wavelength=encoder_wavelengths,
                    GSD=gsd,
                )
                # feats: [B,C,H,W] or list of such → take last
                if isinstance(feats, (list, tuple)):
                    feats = feats[-1]
                if feats.dim() != 4:
                    raise ValueError(f"Expected [B,C,H,W] from encoder, got {feats.shape}")

            feats = feats.detach().cpu()  # [B,C,H,W]
            B, C, H, W = feats.shape

            for i in range(B):
                img_name = Path(paths[i]).name  # e.g. "sample123.pt"
                lbl = int(labels[i])
                fmap = feats[i]  # [C,H,W]

                patches = pc_utils.flatten_feature_map(fmap)  # [H*W, C]
                fs = pc_utils.FeatureSet(
                    image_name=img_name,
                    patches=patches,
                    spatial_size=(H, W),
                )

                label_map[img_name] = lbl
                if lbl == 0:
                    normal_sets.append(fs)
                else:
                    anomalous_sets.append(fs)

    # Sort normals by image_name to get deterministic alternating split
    normal_sets.sort(key=lambda fs: fs.image_name)

    # Alternating split: even indices → memory, odd → normal test
    memory_sets = normal_sets[::2]
    normal_test_sets = normal_sets[1::2]

    print(f"[PatchCore] Normal val samples: {len(normal_sets)} "
          f"(memory: {len(memory_sets)}, normal test: {len(normal_test_sets)})")
    print(f"[PatchCore] Anomalous val samples: {len(anomalous_sets)}")

    # Build memory bank
    bank = pc_memory_bank.MemoryBank(
        sampling_ratio=PATCHCORE_SAMPLING_RATIO,
        coreset_method=PATCHCORE_CORESET_METHOD,
        device=device,
    )
    bank.build(memory_sets)

    # Score normal_test + anomalous
    full_test_set = normal_test_sets + anomalous_sets
    image_scores, _ = bank.compute_anomaly_scores(full_test_set, reduction="mean")

    # Collect scores + labels
    scores_list: List[float] = []
    labels_list: List[int] = []

    for name, score in image_scores:
        lbl = label_map[Path(name).name]  # ensure we use stem if changed
        scores_list.append(score)
        labels_list.append(lbl)

    scores_np = np.asarray(scores_list, dtype=np.float32)
    labels_np = np.asarray(labels_list, dtype=np.int64)

    auroc = compute_auroc(scores_np, labels_np)
    best_thresh, best_f1 = best_f1_threshold(scores_np, labels_np)

    preds = scores_np >= best_thresh
    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(len(labels_np), 1)

    return {
        "auroc": auroc,
        "best_f1": best_f1,
        "best_thresh": float(best_thresh),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def run_training(args: argparse.Namespace, batch_size: int) -> None:
    device = select_device(args.device)
    set_global_seed(args.seed)

    data_root = Path(args.data_root)
    splits_root = Path(args.splits_root)
    train_csv = resolve_csv_path(args.train_csv, splits_root)
    val_csv = resolve_csv_path(args.val_csv, splits_root)
    wavelengths = np.load(args.wavelengths_path)

    config = BackboneConfig(dataset_type="floga")
    img_size = config.img_size
    gsd_meters = float(config.gsd_meters)

    train_dataset = FlogaCsvDataset(
        csv_path=train_csv,
        data_root=data_root,
        wavelengths=wavelengths,
        img_size=img_size,
        only_label=0,  # train on normals only
    )
    val_dataset = FlogaCsvDataset(
        csv_path=val_csv,
        data_root=data_root,
        wavelengths=wavelengths,
        img_size=img_size,
        only_label=None,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Teacher and student start from same pretrained weights
    teacher = build_image_encoder(config)
    teacher = load_weights(teacher, Path(args.teacher_ckpt))
    teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    student = build_image_encoder(config)
    student = load_weights(student, Path(args.teacher_ckpt))
    student.to(device)

    freeze_backbone(student)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples:   {len(val_dataset)}")
    print(f"[INFO] Trainable params: {sum(p.numel() for p in trainable_params)}")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] PatchCore config: sampling_ratio={PATCHCORE_SAMPLING_RATIO}, "
          f"coreset_method='{PATCHCORE_CORESET_METHOD}'")

    # ---------- LOAD OR PRECOMPUTE TEACHER CACHE ----------
    cache_path = Path(args.teacher_cache_path)
    teacher_cache: Dict[str, torch.Tensor]
    center_cpu: torch.Tensor

    if args.load_teacher_cache and cache_path.exists():
        print(f"[INFO] Loading teacher cache from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        teacher_cache = data["cache"]
        center_cpu = data["center"]
    else:
        print("[INFO] Teacher cache not loaded (flag off or file missing). Computing cache...")
        teacher_cache, center_cpu = precompute_teacher_cache(
            teacher=teacher,
            loader=train_loader,
            device=device,
            gsd_meters=gsd_meters,
            img_size=img_size,
            use_amp=use_amp,
        )
        if args.save_teacher_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Saving teacher cache to {cache_path}")
            torch.save({"cache": teacher_cache, "center": center_cpu}, cache_path)

    center = center_cpu.to(device)

    # ---- TEACHER BASELINE (PatchCore on VAL) ----
    teacher_metrics = evaluate_patchcore(
        encoder=teacher,
        loader=val_loader,
        device=device,
        gsd_meters=gsd_meters,
        img_size=img_size,
        use_amp=use_amp,
    )
    print(
        "[TEACHER BASELINE - PatchCore] "
        f"auroc={teacher_metrics['auroc']:.4f} "
        f"best_f1={teacher_metrics['best_f1']:.4f} "
        f"acc={teacher_metrics['accuracy']:.4f} "
        f"prec={teacher_metrics['precision']:.4f} "
        f"recall={teacher_metrics['recall']:.4f}"
    )
    print(f"[TEACHER BASELINE - PatchCore] Confusion matrix: {teacher_metrics['confusion_matrix']}")

    # After caching, you *can* move teacher off GPU (cache used in training)
    teacher.to("cpu")
    torch.cuda.empty_cache()

    best_auroc = -float("inf")
    out_dir = Path(args.out_ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "hyperfree_floga_centerdistill_best.pt"

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        loss = train_one_epoch(
            teacher=teacher,  # unused now
            student=student,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            center=center,
            gsd_meters=gsd_meters,
            img_size=img_size,
            use_amp=use_amp,
            scaler=scaler,
            lambda_center=args.lambda_center,
            lambda_distill=args.lambda_distill,
            teacher_cache=teacher_cache,
        )

        # ---- PatchCore evaluation on VAL with current student ----
        metrics = evaluate_patchcore(
            encoder=student,
            loader=val_loader,
            device=device,
            gsd_meters=gsd_meters,
            img_size=img_size,
            use_amp=use_amp,
        )

        auroc = metrics["auroc"]
        if not math.isnan(auroc) and auroc > best_auroc:
            best_auroc = auroc
            torch.save(
                {
                    "backbone_state_dict": student.state_dict(),
                    "center": center.detach().cpu(),  # keep center for compatibility
                },
                best_path,
            )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss={loss:.6f} "
            f"PatchCore_auroc={metrics['auroc']:.4f} "
            f"best_f1={metrics['best_f1']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f}"
        )
        print(f"[Epoch {epoch + 1}] Confusion matrix (PatchCore): {metrics['confusion_matrix']}")

    print(f"[INFO] Best checkpoint saved to {best_path} (PatchCore AUROC = {best_auroc:.4f})")


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size

    while True:
        try:
            run_training(args, batch_size=batch_size)
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            if (
                args.auto_batch
                and "out of memory" in message
                and torch.cuda.is_available()
                and batch_size > args.min_batch_size
            ):
                new_batch = max(args.min_batch_size, batch_size // 2)
                print(
                    f"[WARN] CUDA OOM at batch size {batch_size}. "
                    f"Retrying with batch size {new_batch}."
                )
                batch_size = new_batch
                torch.cuda.empty_cache()
                continue
            raise


if __name__ == "__main__":
    main()


'''

python train_finetune_hyperfree_floga_distill.py   --data-root ../../../../data/datasets/FLOGA   --splits-root "../../../../data/datasets/FLOGA/finetuning data/floga_splits" --teacher-ckpt ckpt/HyperFree-b.pth   --out-ckpt-dir ckpt   --epochs 10   --batch-size 4   --min-batch-size 2   --lr 1e-4   --weight-decay 1e-4   --lambda-center 0.3   --lambda-distill 1.0  --save-teacher-cache

python train_finetune_hyperfree_floga_distill.py   --data-root ../../../../data/datasets/FLOGA   --splits-root "../../../../data/datasets/FLOGA/finetuning data/floga_splits" --teacher-ckpt ckpt/HyperFree-b.pth   --out-ckpt-dir ckpt   --epochs 10   --batch-size 4   --min-batch-size 2   --lr 1e-4   --weight-decay 1e-4   --lambda-center 0.3   --lambda-distill 1.0  --load-teacher-cache

'''
