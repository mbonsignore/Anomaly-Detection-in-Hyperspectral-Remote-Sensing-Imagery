#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from typing import List, Tuple

import rasterio
from tqdm import tqdm

# ============================================================
# CONFIG (relative to this script's folder)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Folder with the three PRISMA/S2 pairs
ROOT_DIR = BASE_DIR / "Acquisizioni_S2_PRISMA"

# Directory containing Sentinel-2 SRF CSVs (S2_B02.csv, ..., S2_B12.csv)
S2_SRF_DIR = BASE_DIR / "S2_SRF"

S2_SRF_FILES = [
    S2_SRF_DIR / "S2_B02.csv",
    S2_SRF_DIR / "S2_B03.csv",
    S2_SRF_DIR / "S2_B04.csv",
    S2_SRF_DIR / "S2_B05.csv",
    S2_SRF_DIR / "S2_B06.csv",
    S2_SRF_DIR / "S2_B07.csv",
    S2_SRF_DIR / "S2_B08.csv",
    S2_SRF_DIR / "S2_B8A.csv",
    S2_SRF_DIR / "S2_B11.csv",
    S2_SRF_DIR / "S2_B12.csv",
]

# ============================================================
# SRF UTILS
# ============================================================

def load_srf_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SRF CSV with columns: wavelength_nm, srf.
    Returns:
        wl : [N] wavelengths in nm
        srf: [N] non-negative SRF values
    """
    if not path.is_file():
        raise FileNotFoundError(f"SRF CSV not found: {path}")

    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
    wl = data[:, 0]
    srf = data[:, 1]

    order = np.argsort(wl)
    wl = wl[order]
    srf = srf[order]

    return wl.astype(np.float32), np.clip(srf.astype(np.float32), 0.0, None)


def interpolate_srf_to_grid(
    lambda_srf: np.ndarray,
    srf: np.ndarray,
    lambda_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate SRF defined on lambda_srf onto a new grid lambda_grid.
    Out-of-range values are set to 0.
    """
    return np.interp(lambda_grid, lambda_srf, srf, left=0.0, right=0.0).astype(np.float32)


def integrate_band_chw(
    img_chw: np.ndarray,
    srf_grid: np.ndarray,
    dl_grid: np.ndarray,
) -> np.ndarray:
    """
    Spectral integration for a single synthesized band.

    Args:
        img_chw: [C,H,W] reflectance
        srf_grid: [C] SRF values on same wavelength grid as img_chw
        dl_grid: [C] wavelength step for integration

    Returns:
        band_hw: [H,W] float32
    """
    if img_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {img_chw.shape}")

    if img_chw.shape[0] != srf_grid.shape[0] or img_chw.shape[0] != dl_grid.shape[0]:
        raise ValueError(
            f"Channel mismatch: img={img_chw.shape[0]}, "
            f"SRF={srf_grid.shape[0]}, dλ={dl_grid.shape[0]}"
        )

    weights = srf_grid * dl_grid  # [C]
    den = np.sum(weights)
    if den <= 0:
        raise ValueError("SRF integral is zero (no overlap with wavelength grid).")

    # img_chw * [C,1,1] -> [C,H,W], sum over C -> [H,W]
    num = (img_chw * weights[:, None, None]).sum(axis=0)
    return (num / den).astype(np.float32)


def load_all_s2_srfs_on_grid(wavelengths_prisma: np.ndarray) -> np.ndarray:
    """
    Load all Sentinel-2 SRFs and interpolate them onto the PRISMA wavelength grid.

    Args:
        wavelengths_prisma: [C] wavelength per PRISMA band (nm)

    Returns:
        srf_prisma: [num_s2_bands, C]
    """
    srfs: List[np.ndarray] = []
    for path in S2_SRF_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing S2 SRF file: {path}")
        wl_srf, srf = load_srf_csv(path)
        srf_grid = interpolate_srf_to_grid(wl_srf, srf, wavelengths_prisma)

        if np.sum(srf_grid) == 0:
            raise RuntimeError(f"No spectral overlap between PRISMA and SRF {path.name}")

        srfs.append(srf_grid)

    return np.stack(srfs, axis=0)  # [num_s2_bands, C]


# ============================================================
# PRISMA IO
# ============================================================

def load_prisma_reflectance_and_wavelengths(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load PRISMA reflectance_coregistered.tif and extract wavelengths from metadata.

    Returns:
        img: [C,H,W] float32 in [0,1]
        wavelengths_nm: [C] float32
        profile: rasterio profile to reuse georeferencing
    """
    with rasterio.open(path) as ds:
        img = ds.read().astype(np.float32)  # [C,H,W]
        profile = ds.profile

        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        # Read wavelengths
        tags_global = ds.tags()
        wavelengths_nm = None

        # Try global tag first
        for key in ["WAVELENGTH", "wavelength", "Wavelength"]:
            if key in tags_global:
                raw = tags_global[key]
                parts = [p for p in raw.replace(";", ",").replace(" ", ",").split(",") if p]
                wavelengths_nm = np.array([float(p) for p in parts], dtype=np.float32)
                break

        # Fallback: per-band tags
        if wavelengths_nm is None:
            wl_list: List[float] = []
            for bidx in range(1, ds.count + 1):
                band_tags = ds.tags(bidx=bidx)
                if "WAVELENGTH" in band_tags:
                    wl_list.append(float(band_tags["WAVELENGTH"]))
                elif "wavelength" in band_tags:
                    wl_list.append(float(band_tags["wavelength"]))
                else:
                    raise RuntimeError(
                        f"No wavelength metadata for band {bidx} in {path.name} "
                        f"(available keys: {list(band_tags.keys())})"
                    )
            wavelengths_nm = np.array(wl_list, dtype=np.float32)

        if wavelengths_nm.shape[0] != img.shape[0]:
            raise RuntimeError(
                f"Mismatch between image bands ({img.shape[0]}) and wavelengths "
                f"({wavelengths_nm.shape[0]}) in {path.name}"
            )

    return img, wavelengths_nm, profile


def find_prisma_file(folder: Path) -> Path:
    """
    Given an acquisition folder, find *reflectance_coregistered.tif.
    """
    prisma = None
    for tif in folder.glob("*.tif"):
        if "reflectance_coregistered" in tif.name.lower():
            prisma = tif
            break

    if prisma is None:
        raise RuntimeError(f"No PRISMA *reflectance_coregistered.tif found in {folder}")

    return prisma


# ============================================================
# MAIN CONVERSION
# ============================================================

def convert_prisma_folder_to_s2_srf(root_dir: Path) -> None:
    """
    For each acquisition folder under root_dir, synthesize a Sentinel-2-like
    10-band image from the PRISMA reflectance cube using Sentinel-2 SRFs.
    The output is written next to the PRISMA file in the same acquisition folder.
    """
    if not root_dir.is_dir():
        raise RuntimeError(f"Root directory {root_dir} does not exist.")

    acquisitions = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if not acquisitions:
        raise RuntimeError(f"No acquisition folders found under {root_dir}")

    print(f"[INFO] Found {len(acquisitions)} acquisition folders under {root_dir.name}")

    for folder in acquisitions:
        print(f"\n=== Acquisition: {folder.name} ===")
        prisma_path = find_prisma_file(folder)
        print(f"[INFO] PRISMA file: {prisma_path.name}")

        img_chw, wavelengths_nm, profile = load_prisma_reflectance_and_wavelengths(prisma_path)
        C, H, W = img_chw.shape
        print(f"[INFO] PRISMA shape: C={C}, H={H}, W={W}")

        # Wavelength step Δλ
        dl = np.gradient(wavelengths_nm).astype(np.float32)  # [C]

        # Load S2 SRFs on PRISMA grid
        s2_srfs = load_all_s2_srfs_on_grid(wavelengths_nm)  # [10, C]
        num_s2_bands = s2_srfs.shape[0]
        print(f"[INFO] Loaded {num_s2_bands} Sentinel-2 SRFs")

        # Integrate all bands
        synth_bands = []
        for i in tqdm(range(num_s2_bands), desc=f"{folder.name} (S2 synth)"):
            band_hw = integrate_band_chw(img_chw, s2_srfs[i], dl)  # [H,W]
            synth_bands.append(band_hw)

        synth_s2 = np.stack(synth_bands, axis=0).astype(np.float32)  # [10,H,W]

        # Prepare output profile: copy georeferencing, change count/dtype
        out_profile = profile.copy()
        out_profile.update(
            count=num_s2_bands,
            dtype="float32",
        )

        # Save in the same acquisition folder, next to the PRISMA file
        out_path = prisma_path.with_name(prisma_path.stem + "_S2synth.tif")
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(synth_s2)

        print(f"[INFO] Written synthetic Sentinel-2 image: {out_path}")

    print("\n✅ PRISMA → Sentinel-2 SRF conversion completed.")


if __name__ == "__main__":
    convert_prisma_folder_to_s2_srf(ROOT_DIR)