"""End-to-end script to extract Copernicus-FM backbone features (HyperFree style)."""

from __future__ import annotations

from pathlib import Path
from time import time
import random
import numpy as np

import torch
from tqdm import tqdm

from config import DEFAULT_CONFIG, build_config, DATASET
from dataloader import create_dataloader
from model import build_image_encoder


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _select_device(preferred: str) -> torch.device:
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def run(config=DEFAULT_CONFIG) -> None:
    device = _select_device(config.device)
    print(f"[INFO] Using device: {device}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(42)

    dataloader = create_dataloader(config)
    dataset_key = config.dataset_type.lower()
    total_batches = len(dataloader)

    model = build_image_encoder(config).to(device)
    model.eval()

    total_samples = 0
    start_time = time()
    first_logged = False

    with torch.no_grad(), tqdm(total=total_batches, desc="Extracting features", unit="batch") as pbar:
        for batch in dataloader:
            if batch is None:
                continue
            images: torch.Tensor = batch["image"].to(device)
            meta_info: torch.Tensor = batch["meta_info"].to(device)
            wave_list: torch.Tensor = batch["wavelengths"].to(device)
            bandwidth: torch.Tensor = batch["bandwidth"].to(device)
            paths = batch["path"]

            feats = model(images, meta_info, wave_list, bandwidth)

            if not first_logged:
                print(f"[INFO] Dataset: {dataset_key}")
                print(f"[INFO] Input batch shape: {tuple(images.shape)}")
                print(f"[INFO] Output feature shape: {tuple(feats.shape)}")
                first_logged = True

            for tensor, path in zip(feats, paths):
                source_path = Path(path)
                if dataset_key in ("spectralearth", "spectralearth7bands"):
                    try:
                        relative_parts = source_path.relative_to(config.data_dir).parts
                    except ValueError:
                        relative_parts = ()
                    folder_name = (
                        relative_parts[0]
                        if len(relative_parts) > 1
                        else source_path.parent.name
                    )
                    filename = f"{folder_name}_{source_path.stem}.pt"
                elif dataset_key == "copernicuspretrain":
                    try:
                        relative_parts = source_path.relative_to(config.data_dir).parts
                    except ValueError:
                        relative_parts = ()
                    if len(relative_parts) >= 3:
                        grid_folder = relative_parts[-3]
                        local_folder = relative_parts[-2]
                        tif_name = Path(relative_parts[-1]).stem
                        filename = f"{grid_folder}__{local_folder}__{tif_name}.pt"
                    else:
                        filename = source_path.stem + ".pt"
                else:
                    filename = source_path.stem + ".pt"

                save_path = config.output_dir / filename
                torch.save(tensor.cpu(), save_path)
                total_samples += 1

            pbar.set_postfix({
                "samples": total_samples,
                "avg_time/batch": f"{(time() - start_time) / (pbar.n + 1):.2f}s"
            })
            pbar.update(1)

    print(f"\n✅ Finished! Saved {total_samples} feature files to {config.output_dir}")


if __name__ == "__main__":
    run(build_config(DATASET))
