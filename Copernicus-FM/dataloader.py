"""Dataset and DataLoader utilities for Copernicus-FM backbone extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import csv

import numpy as np
import torch
import tifffile
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms.functional import InterpolationMode, resize
from torch.utils.data.dataloader import default_collate

from config import BackboneConfig
from utils.spectral_meta import resolve_wave_bandwidth


class Dataset(TorchDataset):
    """Loads multispectral patches with shared preprocessing."""

    def __init__(self, config: BackboneConfig) -> None:
        self.config = config
        self.dataset_type = config.dataset_type.lower()

        if self.dataset_type == "floga":
            self.root_dir = None
        else:
            self.root_dir = Path(config.data_dir)
            if not self.root_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        if self.dataset_type in ("activefire", "hyperseg", "hyperseg8bands"):
            self.image_paths = sorted(self.root_dir.glob("*.tif"))
        elif self.dataset_type in ("spectralearth", "spectralearth7bands"):
            self.image_paths = self._gather_spectralearth_tiles()
        elif self.dataset_type == "copernicuspretrain":
            self.image_paths = self._gather_copernicus_tiles()
        elif self.dataset_type in ("flogapre", "flogapost"):
            self.image_paths = sorted(self.root_dir.glob("*.pt"))
        elif self.dataset_type == "floga":
            if config.data_dirs is None:
                raise ValueError("dataset_type='floga' requires config.data_dirs")

            # Common parent, e.g. .../FLOGA
            base_root = Path(config.data_dirs[0]).parent

            print(f"[INFO] dataset_type='floga', images_csv={config.images_csv}")

            if config.images_csv is not None:
                csv_path = Path(config.images_csv)
                if not csv_path.is_file():
                    raise FileNotFoundError(f"FLOGA CSV not found: {csv_path}")

                print(f"[INFO] Loading FLOGA paths from CSV: {csv_path}")
                image_paths: List[Path] = []

                with csv_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    expected = {"sample_name", "folder", "label"}
                    if not expected.issubset(reader.fieldnames or []):
                        raise ValueError(
                            f"CSV {csv_path} must contain columns {sorted(expected)}, "
                            f"found {reader.fieldnames}"
                        )

                    for row in reader:
                        sample_name = row["sample_name"]   # includes _pre / _post
                        folder = row["folder"]             # e.g. "FLOGA_PRE" or "FLOGA_POST"
                        path = base_root / folder / f"{sample_name}.pt"
                        if not path.is_file():
                            print(f"[WARN] File listed in CSV not found on disk: {path}")
                            continue
                        image_paths.append(path)

                self.image_paths = sorted(image_paths)
                if not self.image_paths:
                    raise FileNotFoundError(
                        f"No valid .pt files found from CSV {csv_path} under base {base_root}"
                    )
            else:
                # Fallback: scan both dirs
                all_paths = []
                print("[INFO] No images_csv provided; using ALL .pt files in FLOGA_PRE and FLOGA_POST.")
                for d in config.data_dirs:
                    d = Path(d)
                    if not d.exists():
                        raise FileNotFoundError(f"FLOGA directory not found: {d}")
                    all_paths.extend(sorted(d.glob("*.pt")))
                self.image_paths = sorted(all_paths)

        if not self.image_paths:
            raise FileNotFoundError(
                f"No image files discovered for dataset '{config.dataset_type}'"
            )

        self.wave_cache: torch.Tensor | None = None
        self.band_cache: torch.Tensor | None = None
        # Preload wavelengths from file if provided (HyperFree behavior)
        if config.wavelengths_path is not None and Path(config.wavelengths_path).exists():
            waves_np = np.load(config.wavelengths_path)
            self.wave_cache = torch.as_tensor(waves_np, dtype=torch.float32)

    def _gather_copernicus_tiles(self) -> List[Path]:
        tiles = []
        for local_folder in self.root_dir.glob("*/*"):
            if local_folder.is_dir():
                tif_files = sorted(local_folder.glob("*.tif"))
                if tif_files:
                    tiles.append(tif_files[0])
        return tiles

    def _gather_spectralearth_tiles(self) -> List[Path]:
        return sorted(
            tif_path
            for tif_path in self.root_dir.rglob("*.tif")
            if tif_path.is_file()
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resolve_meta(self, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.wave_cache is None or self.band_cache is None:
            self.wave_cache, self.band_cache = resolve_wave_bandwidth(
                self.dataset_type, channels, self.config.wavelengths_path, self.config.bandwidth_path
            )
        return self.wave_cache, self.band_cache

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        path = self.image_paths[index]

        if self.dataset_type in ("flogapre", "flogapost", "floga"):
            tensor = torch.load(path, map_location="cpu").float()
            if tensor.ndim != 3:
                raise ValueError(f"{path} has invalid shape {tensor.shape}")
        else:
            try:
                array = tifffile.imread(path).astype(np.float32)
            except Exception as e:
                # log and skip
                print(f"[WARN] Failed reading {path}: {repr(e)}")
                return None
            if array.ndim == 2:
                array = array[..., None]
            elif array.ndim == 3 and array.shape[0] < array.shape[2]:
                array = np.moveaxis(array, 0, -1)
            tensor = torch.from_numpy(array).permute(2, 0, 1)

        original_shape = torch.tensor(tensor.shape[-2:], dtype=torch.int)

        tensor = resize(
            tensor,
            size=[self.config.img_size, self.config.img_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        for c in range(tensor.shape[0]):
            band = tensor[c]
            bmin = band.min()
            bmax = band.max()
            if (bmax - bmin) > 0:
                tensor[c] = (band - bmin) / (bmax - bmin)

        wave_list, bandwidth = self._resolve_meta(tensor.shape[0])
        meta_info = torch.full((4,), float("nan"), dtype=torch.float32)

        return {
            "image": tensor,
            "wavelengths": wave_list.clone(),
            "bandwidth": bandwidth.clone(),
            "meta_info": meta_info,
            "path": str(path),
            "original_shape": original_shape,
        }
    

def collate_skip_none(batch):
    """
    Filters out samples that are None (e.g., corrupted/unreadable files).
    Returns None if the whole batch is invalid.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def worker_init_fn(worker_id: int):
    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def create_dataloader(config: BackboneConfig) -> DataLoader:
    dataset = Dataset(config)

    if config.dataset_type.lower() == "floga":
        source_info = f"FLOGA dirs={config.data_dirs}"
        if config.images_csv is not None:
            source_info += f", csv={config.images_csv}"
    else:
        source_info = str(config.data_dir)

    print(
        f"Preparing dataset '{config.dataset_type}' from {source_info} | tiles: {len(dataset)}"
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_skip_none,
        drop_last=False,
    )


__all__ = ["Dataset", "create_dataloader"]
