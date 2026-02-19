"""Static configuration for Copernicus-FM backbone extraction (HyperFree style)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from utils.spectral_meta import get_spectral_metadata

# Dataset selector
DATASET = "floga"

# Default output directory (matches HyperFree style)
OUTPUT_DIR = Path("Extracted Features/features")

# Resolve spectral metadata once (no external files)
WAVELENGTHS, BANDWIDTHS = get_spectral_metadata(DATASET)


@dataclass
class BackboneConfig:
    dataset_type: str
    data_dir: Optional[Path] = None
    data_dirs: Optional[List[Path]] = None

    # NEW: Optional CSV listing which FLOGA samples to process (train/val/test split)
    images_csv: Optional[Path] = None

    gsd_meters: Optional[float] = None
    wavelengths_path: Optional[Path] = None
    bandwidth_path: Optional[Path] = None

    checkpoint_path: Path = Path("ckpt/copernicus_floga_semisupervised_best_fourthrun.pt") #CopernicusFM_ViT_base_varlang_e100.pth
    output_dir: Path = OUTPUT_DIR

    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False

    img_size: int = 224
    vit_patch_size: int = 16
    device: str = "cuda:0"

    input_mode: str = "spectral"
    kernel_size: int = 16
    feature_block_index: int = 11
    output_grid_size: str = "4x4"  # "native" or "4x4"


# Dataset presets (paths and metadata)
_DATASET_DEFAULTS = {
    "activefire": {
        "data_dir": Path("../ActiveFire/Asia_NoThermal"),
        "gsd_meters": 30.0,
    },
    "hyperseg": {
        "data_dir": Path("../HyperSeg/HyperSeg_Full"),
        "gsd_meters": 3.3,
    },
    "hyperseg8bands": {
        "data_dir": Path("../HyperSeg/HyperSeg_8bands"),
        "gsd_meters": 3.3,
    },
    "spectralearth": {
        "data_dir": Path("../spectralearth/enmap_subset"),
        "gsd_meters": 30.0,
    },
    "spectralearth7bands": {
        "data_dir": Path("../SpectralEarth/enmap_subset_7bands"),
        "gsd_meters": 30.0,
    },
    "copernicuspretrain": {
        "data_dir": Path("../CopernicusPretrain/raw_geotiffs_220k_aligned/s2_toa_mix/images"),
        "gsd_meters": 10.0,
    },
    "flogapre": {
        "data_dir": Path("../FLOGA/FLOGA_PRE"),
        "gsd_meters": 20.0,
    },
    "flogapost": {
        "data_dir": Path("../FLOGA/FLOGA_POST"),
        "gsd_meters": 20.0,
    },
    "floga": {
        "data_dirs": [
            Path("../FLOGA/FLOGA_PRE"),
            Path("../FLOGA/FLOGA_POST"),
        ],
        "gsd_meters": 20.0,

        # NEW: default CSV (change to train/val/test as needed)
        "images_csv": Path("../FLOGA/finetuning data/floga_splits/images_test_semisupervised.csv"),
    },
}


def build_config(dataset: str = DATASET) -> BackboneConfig:
    key = dataset.lower()
    if key not in _DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset '{dataset}'")
    defaults = _DATASET_DEFAULTS[key]

    if key == "floga":
        return BackboneConfig(
            dataset_type=key,
            data_dir=None,
            data_dirs=defaults["data_dirs"],
            images_csv=defaults.get("images_csv", None),
            gsd_meters=defaults["gsd_meters"],
            wavelengths_path=None,
            bandwidth_path=None,
        )

    return BackboneConfig(
        dataset_type=key,
        data_dir=defaults["data_dir"],
        data_dirs=None,
        images_csv=None,
        gsd_meters=defaults["gsd_meters"],
        wavelengths_path=None,
        bandwidth_path=None,
    )


DEFAULT_CONFIG = build_config()