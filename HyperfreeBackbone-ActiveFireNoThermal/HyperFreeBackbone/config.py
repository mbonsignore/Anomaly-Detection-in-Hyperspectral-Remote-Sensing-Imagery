"""Configuration values for the standalone HyperFree backbone extractor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BackboneConfig:

    dataset_type: str = "floga"

    # Single-folder mode (all legacy datasets)
    data_dir: Path | None = None

    # Multi-folder mode (only for floga)
    data_dirs: list[Path] | None = None

    # Optional: CSV listing which FLOGA samples to process
    images_csv: Path | None = None

    wavelengths_path: Path | None = None
    gsd_meters: float | None = None

    checkpoint_path: Path = field(
        default_factory=lambda: Path(

            #hyperfree_floga_semisupervised_best_firstrun.pt    hyperfree_floga_semisupervised_best_secondrun.pt    HyperFree-b.pth     hyperfree_floga_centerloss_best.pt

            os.getenv("HYPERFREE_BACKBONE_CKPT", "ckpt/hyperfree_floga_semisupervised_best_fourthrun.pt")      
        )
    )
    output_dir: Path = Path("../../../../data/datasets/hyperfree/Extracted Features/features")

    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = False

    img_size: int = 1024
    vit_patch_size: int = 16
    gsd_meters: float | None = None

    device: str = "cuda:0"

    def __post_init__(self) -> None:
        dataset_defaults = {
            "activefire": {
                "data_dir": Path("../../../../data/datasets/ActiveFire/Asia_NoThermal"),
                "wavelengths": Path("../wavelengths_landsat8_no_thermal.npy"),
                "gsd_meters": 30.0,
            },
            "hyperseg": {
                "data_dir": Path("../../../../data/datasets/HyperSeg/HyperSeg_Full"),
                "wavelengths": Path("../wavelengths_hyperseg.npy"),
                "gsd_meters": 3.3,
            },
            "hyperseg8bands": {
                "data_dir": Path("../../../../data/datasets/HyperSeg/HyperSeg_8bands"),
                "wavelengths": Path("../wavelengths_landsat8_no_thermal.npy"),
                "gsd_meters": 3.3,
            },
            "spectralearth": {
                "data_dir": Path("../../../../vandal/datasets/spectralearth/enmap_subset"),
                "wavelengths": Path("../wavelengths_spectralearth.npy"),
                "gsd_meters": 30.0,
            },
            "spectralearth7bands": {
                "data_dir": Path("../../../../data/datasets/SpectralEarth/enmap_subset_7bands_srf"),
                "wavelengths": Path("../wavelengths_spectralearth_7bands.npy"),
                "gsd_meters": 30.0,
            },
            "copernicuspretrain": {
                "data_dir": Path("../../../../data/datasets/CopernicusPretrain/raw_geotiffs_220k_aligned/s2_toa_mix/images"),
                "wavelengths": Path("../wavelengths_copernicus.npy"),
                "gsd_meters": 10.0,
            },
            "flogapre": {
                "data_dir": Path("../../../../data/datasets/FLOGA/FLOGA_PRE"),
                "wavelengths": Path("../wavelengths_floga.npy"),
                "gsd_meters": 20.0,
            },
            "flogapost": {
                "data_dir": Path("../../../../data/datasets/FLOGA/FLOGA_POST"),
                "wavelengths": Path("../wavelengths_floga.npy"),
                "gsd_meters": 20.0,
            },
            # ---- NEW: floga merges both folders ----
            "floga": {
                "data_dirs": [
                    Path("../../../../data/datasets/FLOGA/FLOGA_PRE"),
                    Path("../../../../data/datasets/FLOGA/FLOGA_POST"),
                ],
                "wavelengths": Path("../wavelengths_floga.npy"),
                "gsd_meters": 20.0,
                "images_csv": Path("../../../../data/datasets/FLOGA/finetuning data/floga_splits/images_test_semisupervised.csv"),
            },
        }

        key = self.dataset_type.lower()
        if key not in dataset_defaults:
            raise ValueError(f"Unsupported dataset_type '{self.dataset_type}'")

        defaults = dataset_defaults[key]

        # ---------------------------------------------------------
        # SPECIAL HANDLING FOR "floga" → uses multi-directory mode
        # ---------------------------------------------------------
        if key == "floga":
            # If user passed data_dirs manually, use them
            if self.data_dirs is not None:
                self.data_dirs = [Path(p) for p in self.data_dirs]
            else:
                # Otherwise use defaults
                self.data_dirs = defaults["data_dirs"]

            self.data_dir = None  # disable single-dir mode completely

        # ---------------------------------------------------------
        # ALL OTHER DATASETS → legacy single-directory mode
        # ---------------------------------------------------------
        else:
            if self.data_dir is None:
                self.data_dir = defaults["data_dir"]
            else:
                self.data_dir = Path(self.data_dir)

            self.data_dirs = None  # ensure multi-folder mode off

        # ---------------------------------------------------------
        # Shared default handling
        # ---------------------------------------------------------
        if self.wavelengths_path is None:
            self.wavelengths_path = defaults["wavelengths"]
        else:
            self.wavelengths_path = Path(self.wavelengths_path)

        if self.gsd_meters is None:
            self.gsd_meters = defaults["gsd_meters"]
        else:
            self.gsd_meters = float(self.gsd_meters)
        
        if self.images_csv is None:
            self.images_csv = defaults["images_csv"] if "images_csv" in defaults else None
        elif self.images_csv is not None:
            self.images_csv = Path(self.images_csv)


DEFAULT_CONFIG = BackboneConfig()