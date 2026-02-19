"""Dataset and DataLoader utilities shared by ActiveFire, HyperSeg,
SpectralEarth, CopernicusPretrain, and FLOGA inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import tifffile
import csv
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm  # NEW

from config import BackboneConfig


class Dataset(TorchDataset):
    """Loads hyperspectral/multispectral patches with shared preprocessing.

    Supports:
        - .tif tiles (ActiveFire, HyperSeg, SpectralEarth, Copernicus)
        - .pt  tiles (FLOGA_PRE, FLOGA_POST, FLOGA)
    """

    def __init__(self, config: BackboneConfig, wavelengths: np.ndarray) -> None:
        self.dataset_type = config.dataset_type.lower()

        # -------------------------------
        # floga = special multi-folder mode
        # -------------------------------
        if self.dataset_type == "floga":
            self.root_dir = None  # avoid Path(None)
        else:
            self.root_dir = Path(config.data_dir)
            if not self.root_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        # ---------------------------------------------------------
        # Dataset type → input file type
        # ---------------------------------------------------------
        if self.dataset_type == "activefire":
            self.image_paths = sorted(self.root_dir.glob("*.tif"))

        elif self.dataset_type == "hyperseg":
            self.image_paths = sorted(self.root_dir.glob("*.tif"))

        elif self.dataset_type == "hyperseg8bands":
            self.image_paths = sorted(self.root_dir.glob("*.tif"))

        elif self.dataset_type == "spectralearth":
            self.image_paths = self._gather_spectralearth_tiles()

        elif self.dataset_type == "spectralearth7bands":
            self.image_paths = self._gather_spectralearth_tiles()

        elif self.dataset_type == "copernicuspretrain":
            self.image_paths = self._gather_copernicus_tiles()

        elif self.dataset_type == "flogapre":
            self.image_paths = sorted(self.root_dir.glob("*.pt"))

        elif self.dataset_type == "flogapost":
            self.image_paths = sorted(self.root_dir.glob("*.pt"))

        elif self.dataset_type == "floga":
            # Multi-directory loader with optional CSV filtering
            all_paths: List[Path] = []
            if config.data_dirs is None:
                raise ValueError("dataset_type='floga' requires config.data_dirs")

            # Common parent, e.g. .../FLOGA
            base_root = Path(config.data_dirs[0]).parent

            # DEBUG: show what config.images_csv is
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
                        # label = int(row["label"])        # not needed here

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
                # Fallback: old behavior, use all .pt in FLOGA_PRE and FLOGA_POST
                print("[INFO] No images_csv provided; using ALL .pt files in FLOGA_PRE and FLOGA_POST.")
                for d in config.data_dirs:
                    d = Path(d)
                    if not d.exists():
                        raise FileNotFoundError(f"FLOGA directory not found: {d}")
                    all_paths.extend(sorted(d.glob("*.pt")))
                self.image_paths = sorted(all_paths)

        else:
            raise ValueError(f"Unsupported dataset_type '{config.dataset_type}'.")

        if not self.image_paths:
            raise FileNotFoundError(
                f"No image files discovered for dataset '{config.dataset_type}'"
            )

        self.wavelengths = torch.as_tensor(wavelengths, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # Dataset gathering helpers
    # -------------------------------------------------------------------------
    def _gather_copernicus_tiles(self) -> List[Path]:
        """Gather exactly one .tif from each localId_lon_lat subfolder."""
        tiles: List[Path] = []

        # First collect all local folders (grid/localId_lon_lat)
        local_folders = [
            f for f in self.root_dir.glob("*/*") if f.is_dir()
        ]

        # Wrap with tqdm to see progress while scanning many folders
        for local_folder in tqdm(
            local_folders,
            desc=f"Scanning Copernicus tiles in {self.root_dir}",
            unit="folder",
        ):
            tif_files = sorted(local_folder.glob("*.tif"))
            if tif_files:
                tiles.append(tif_files[0])
        return tiles

    def _gather_spectralearth_tiles(self) -> List[Path]:
        """Recursively gather tiles nested inside per-scene subdirectories."""
        tiles: List[Path] = []

        # rglob over big trees can be slow → monitor with tqdm
        for tif_path in tqdm(
            self.root_dir.rglob("*.tif"),
            desc=f"Scanning SpectralEarth tiles in {self.root_dir}",
            unit="file",
        ):
            if tif_path.is_file():
                tiles.append(tif_path)

        return sorted(tiles)

    # -------------------------------------------------------------------------
    # Core dataset interface
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        path = self.image_paths[index]

        # ---------------------------------------------------------------------
        # CASE 1 — FLOGA (.pt input)
        # ---------------------------------------------------------------------
        if self.dataset_type in ("flogapre", "flogapost", "floga"):
            try:
                tensor = torch.load(path, map_location="cpu").float()
            except Exception as e:
                print(f"❌ Failed to load PT file: {path}   ↳ {e}")
                return self.__getitem__((index + 1) % len(self.image_paths))

            # Expect shape (C,H,W)
            if tensor.ndim != 3:
                raise ValueError(f"{path} has invalid shape {tensor.shape}")

            original_shape = torch.tensor(tensor.shape[-2:], dtype=torch.int)

            # Resize to 1024×1024
            tensor = resize(
                tensor,
                size=[1024, 1024],
                interpolation=InterpolationMode.BILINEAR,
            )

            # Normalize channels to [0,1]
            for c in range(tensor.shape[0]):
                band = tensor[c]
                bmin = band.min()
                bmax = band.max()
                if (bmax - bmin) > 0:
                    tensor[c] = (band - bmin) / (bmax - bmin)

            return {
                "image": tensor,
                "wavelengths": self.wavelengths.clone(),
                "path": str(path),
                "original_shape": original_shape,
            }

        # ---------------------------------------------------------------------
        # CASE 2 — TIFF datasets (original behavior)
        # ---------------------------------------------------------------------
        try:
            array = tifffile.imread(path).astype(np.float32)
        except Exception as e:
            print(f"❌ Skipping corrupted TIFF: {path}\n   ↳ {e}")
            next_index = (index + 1) % len(self.image_paths)
            return self.__getitem__(next_index)

        # Fix array shape to (H,W,C)
        if array.ndim == 2:  # single band
            array = array[..., None]

        elif array.ndim == 3 and array.shape[0] < array.shape[2]:
            # Convert CHW → HWC (rare TIFF case)
            array = np.moveaxis(array, 0, -1)

        # Convert to tensor C,H,W
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        original_shape = torch.tensor(tensor.shape[-2:], dtype=torch.int)

        # Resize TIFF to 1024×1024
        tensor = resize(
            tensor,
            size=[1024, 1024],
            interpolation=InterpolationMode.BILINEAR,
        )

        # Per-channel min-max normalization
        for c in range(tensor.shape[0]):
            band = tensor[c]
            bmin = band.min()
            bmax = band.max()
            if (bmax - bmin) > 0:
                tensor[c] = (band - bmin) / (bmax - bmin)

        return {
            "image": tensor,
            "wavelengths": self.wavelengths.clone(),
            "path": str(path),
            "original_shape": original_shape,
        }


# -----------------------------------------------------------------------------
# Top-level factory
# -----------------------------------------------------------------------------
def create_dataloader(config: BackboneConfig) -> DataLoader:
    wavelengths = np.load(config.wavelengths_path)
    dataset = Dataset(config, wavelengths)

    if config.dataset_type.lower() == "floga":
        source_info = f"FLOGA dirs={config.data_dirs}"
        if config.images_csv is not None:
            source_info += f", csv={config.images_csv}"
    else:
        source_info = str(config.data_dir)

    print(
        f"Preparing dataset '{config.dataset_type}' from {source_info} | "
        f"tiles: {len(dataset)} | wavelengths shape: {wavelengths.shape}"
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


__all__ = ["Dataset", "create_dataloader"]