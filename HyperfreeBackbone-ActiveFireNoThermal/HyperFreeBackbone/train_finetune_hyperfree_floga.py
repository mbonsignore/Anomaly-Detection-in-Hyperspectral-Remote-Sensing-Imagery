#!/usr/bin/env python3
"""Fine-tune the HyperFree backbone on FLOGA with center-based anomaly learning."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode, resize

from config import BackboneConfig
from model import build_image_encoder, load_weights
from utils.common import LayerNorm2d


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
                    f"CSV {csv_path} must include columns {sorted(expected)}; found {reader.fieldnames}"
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

        # Per-channel min-max normalization.
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
        description="Fine-tune HyperFree backbone on FLOGA with a center loss objective."
    )
    parser.add_argument("--data-root", type=str, default=str(default_data_root))
    parser.add_argument("--splits-root", type=str, default=str(default_splits_root))
    parser.add_argument("--train-csv", type=str, default="images_train.csv")
    parser.add_argument("--val-csv", type=str, default="images_val.csv")
    parser.add_argument("--wavelengths-path", type=str, default=str(base_dir / "../wavelengths_floga.npy"))
    parser.add_argument("--pretrained-ckpt", type=str, default=str(base_dir / "ckpt/HyperFree-b.pth"))
    parser.add_argument("--out-ckpt-dir", type=str, default=str(base_dir / "ckpt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    parser.add_argument("--center-momentum", type=float, default=0.0)
    parser.add_argument("--feature-norm-weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--auto-batch", action="store_true", default=True)
    parser.add_argument("--no-auto-batch", dest="auto_batch", action="store_false")
    parser.add_argument("--min-batch-size", type=int, default=8)
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
    """Return a single embedding per sample using the final patch tokens."""
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


def compute_center(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gsd_meters: float,
    img_size: int,
    use_amp: bool,
) -> torch.Tensor:
    encoder.eval()
    sum_embed = None
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing center", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            batch_wavelengths = batch["wavelengths"]
            encoder_wavelengths = batch_wavelengths[0].tolist()
            gsd = compute_gsd(batch["original_shape"], gsd_meters, img_size)

            with torch.cuda.amp.autocast(enabled=use_amp):
                features = encoder(
                    images,
                    test_mode=True,
                    input_wavelength=encoder_wavelengths,
                    GSD=gsd,
                )
                z = get_embedding(features).float()

            batch_sum = z.sum(dim=0)
            sum_embed = batch_sum if sum_embed is None else sum_embed + batch_sum
            total += z.shape[0]

    if total == 0:
        raise RuntimeError("Cannot initialize center: no samples in loader.")

    return sum_embed / float(total)


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
    encoder: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    center: torch.Tensor,
    gsd_meters: float,
    img_size: int,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    center_momentum: float,
    feature_norm_weight: float,
) -> Tuple[float, torch.Tensor]:
    encoder.train()
    running_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        labels = batch["label"]
        if labels.max().item() != 0:
            raise ValueError("Training batch contains anomalous samples (label=1).")

        images = batch["image"].to(device, non_blocking=True)
        batch_wavelengths = batch["wavelengths"]
        encoder_wavelengths = batch_wavelengths[0].tolist()
        gsd = compute_gsd(batch["original_shape"], gsd_meters, img_size)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            features = encoder(
                images,
                test_mode=True,
                input_wavelength=encoder_wavelengths,
                GSD=gsd,
            )
            z = get_embedding(features)
            diff = z - center.to(z.dtype)
            loss = diff.pow(2).sum(dim=1).mean()
            if feature_norm_weight > 0:
                loss = loss + feature_norm_weight * z.pow(2).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if center_momentum > 0:
            with torch.no_grad():
                batch_mean = z.detach().float().mean(dim=0)
                center.mul_(center_momentum).add_(batch_mean, alpha=1.0 - center_momentum)

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, center


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
            if labels_sorted[j] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
            j += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = score

        i = j

    return best_thresh, best_f1


def evaluate(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    center: torch.Tensor,
    gsd_meters: float,
    img_size: int,
    use_amp: bool,
) -> Dict[str, float | List[List[int]]]:
    encoder.eval()
    scores: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            batch_wavelengths = batch["wavelengths"]
            encoder_wavelengths = batch_wavelengths[0].tolist()
            gsd = compute_gsd(batch["original_shape"], gsd_meters, img_size)

            with torch.cuda.amp.autocast(enabled=use_amp):
                features = encoder(
                    images,
                    test_mode=True,
                    input_wavelength=encoder_wavelengths,
                    GSD=gsd,
                )
                z = get_embedding(features)
                dist = torch.norm(z - center.to(z.dtype), dim=1)

            scores.extend(dist.detach().float().cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())

    scores_np = np.asarray(scores, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)

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
    config.checkpoint_path = Path(args.pretrained_ckpt)
    img_size = config.img_size
    gsd_meters = float(config.gsd_meters)

    train_dataset = FlogaCsvDataset(
        csv_path=train_csv,
        data_root=data_root,
        wavelengths=wavelengths,
        img_size=img_size,
        only_label=0,
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

    encoder = build_image_encoder(config)
    encoder = load_weights(encoder, config.checkpoint_path)
    encoder.to(device)

    freeze_backbone(encoder)

    trainable_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples: {len(val_dataset)}")
    print(f"[INFO] Trainable params: {sum(p.numel() for p in trainable_params)}")
    print(f"[INFO] Using device: {device}")

    # Initialize center from the first pass over training data.
    center = compute_center(
        encoder=encoder,
        loader=train_loader,
        device=device,
        gsd_meters=gsd_meters,
        img_size=img_size,
        use_amp=use_amp,
    ).to(device)

    best_auroc = -float("inf")
    out_dir = Path(args.out_ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "hyperfree_floga_centerloss_best.pt"

    for epoch in trange(args.epochs, desc="Epochs"):
        loss, center = train_one_epoch(
            encoder=encoder,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            center=center,
            gsd_meters=gsd_meters,
            img_size=img_size,
            use_amp=use_amp,
            scaler=scaler,
            center_momentum=args.center_momentum,
            feature_norm_weight=args.feature_norm_weight,
        )

        metrics = evaluate(
            encoder=encoder,
            loader=val_loader,
            device=device,
            center=center,
            gsd_meters=gsd_meters,
            img_size=img_size,
            use_amp=use_amp,
        )

        auroc = metrics["auroc"]
        if not math.isnan(auroc) and auroc > best_auroc:
            best_auroc = auroc
            torch.save(
                {
                    "backbone_state_dict": encoder.state_dict(),
                    "center": center.detach().cpu(),
                },
                best_path,
            )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss={loss:.6f} "
            f"auroc={metrics['auroc']:.4f} "
            f"best_f1={metrics['best_f1']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f}"
        )
        print(f"[Epoch {epoch + 1}] Confusion matrix: {metrics['confusion_matrix']}")

    print(f"[INFO] Best checkpoint saved to {best_path}")


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

python3 train_finetune_hyperfree_floga.py --data-root ../../../../data/datasets/FLOGA --splits-root "../../../../data/datasets/FLOGA/finetuning data/floga_splits" --train-csv images_train.csv --val-csv images_val.csv --wavelengths-path ../wavelengths_floga.npy --pretrained-ckpt ckpt/HyperFree-b.pth --out-ckpt-dir ckpt --device cuda:0 --epochs 50 --batch-size 16 --min-batch-size 1 --auto-batch  --num-workers 4 --amp

'''