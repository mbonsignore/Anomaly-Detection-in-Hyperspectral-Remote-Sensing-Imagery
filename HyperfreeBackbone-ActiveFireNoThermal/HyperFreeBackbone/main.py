"""End-to-end script that reads Landsat patches and saves HyperFree encoder features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from tqdm import tqdm
from time import time
import random
import numpy as np

import torch

from config import BackboneConfig, DEFAULT_CONFIG
from dataloader import create_dataloader
from model import build_image_encoder, load_weights

def set_global_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
        print("[INFO] Enabled full deterministic algorithms in PyTorch.")
    except Exception as e:
        print(f"[WARN] Could not enable full deterministic algorithms: {e}")

def _select_device(preferred: str) -> torch.device:
    """Gracefully fall back to CPU when CUDA is not available."""

    if preferred == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def run(config: BackboneConfig = DEFAULT_CONFIG) -> None:
    device = _select_device(config.device)
    print(f"Using device: {device}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(42)

    dataloader = create_dataloader(config)
    dataset_key = config.dataset_type.lower()
    total_batches = len(dataloader)

    encoder = build_image_encoder(config)
    encoder = load_weights(encoder, config.checkpoint_path)
    encoder.to(device)
    encoder.eval()

    total_samples = 0
    start_time = time()

    with torch.no_grad(), tqdm(total=total_batches, desc="Extracting features", unit="batch") as pbar:
        for batch in dataloader:
            images: torch.Tensor = batch["image"].to(device)
            batch_wavelengths: torch.Tensor = batch["wavelengths"]
            paths = batch["path"]
            encoder_wavelengths = batch_wavelengths[0].tolist()
            original_shapes = batch["original_shape"]
            target_img_size = config.img_size

            gsd = [
                config.gsd_meters * (orig_w.item() / target_img_size)
                for (_, orig_w) in original_shapes
            ]

            multi_scale_feats = encoder(
                images,
                test_mode=True,
                input_wavelength=encoder_wavelengths,
                GSD=gsd,
            )

            features = multi_scale_feats[-1]

            for tensor, path in zip(features, paths):
                source_path = Path(path)
                if dataset_key == "spectralearth" or dataset_key == "spectralearth7bands":
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
                        # fallback in case folder depth is shorter than expected
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
    run()
