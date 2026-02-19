#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import rasterio
import torch.nn.functional as F

from config import BackboneConfig
from model import build_image_encoder, load_weights

# ================================
# CONFIG
# ================================

ROOT_DIR = Path("../../../../data/datasets/GMATICS/Acquisizioni_S2_PRISMA")

HYPERFREE_CKPT = Path("ckpt/HyperFree-b.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentinel-2: all bands resampled to 10m
GSD_S2 = 10.0
# PRISMA: typical ~30m (adjust if you know the exact value)
GSD_PRISMA = 30.0

# Sentinel-2 central wavelengths in nm, in order:
# B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
S2_WAVELENGTHS_NM = np.array([
    492.4,   # B02 (blue)
    559.8,   # B03 (green)
    664.6,   # B04 (red)
    704.1,   # B05 (red edge)
    740.5,   # B06 (red edge)
    782.8,   # B07 (red edge)
    832.8,   # B08 (NIR)
    864.7,   # B8A (NIR)
    1613.7,  # B11 (SWIR)
    2202.4,  # B12 (SWIR)
], dtype=np.float32)


# ================================
# UTILS
# ================================

def get_hyperfree_encoder() -> Tuple[torch.nn.Module, int]:
    """
    Build HyperFree image encoder and return (encoder, img_size).
    Uses the same config API you used in finetuning.
    """
    config = BackboneConfig(dataset_type="floga")  # or your preferred config
    img_size = config.img_size

    encoder = build_image_encoder(config)
    encoder = load_weights(encoder, HYPERFREE_CKPT)
    encoder.to(DEVICE)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, img_size


def get_embedding_from_feature_map(features: torch.Tensor) -> torch.Tensor:
    """
    Average over spatial tokens of the last feature map [B,C,H,W] → [B,D].
    """
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if features.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(features.shape)}")
    return features.flatten(2).mean(dim=2)  # [B,D]


def load_sentinel2_reflectance(path: Path) -> np.ndarray:
    """
    Load sentinel_mosaic.tif and convert DN -> reflectance.
    Returns array [C,H,W] in float32, clipped to [0,1].
    """
    with rasterio.open(path) as ds:
        # rasterio reads [bands, H, W] by default
        dn = ds.read().astype(np.float32)

    refl = (dn - 1000.0) / 10000.0
    refl = np.clip(refl, 0.0, 1.0).astype(np.float32)
    return refl  # [C,H,W]


def load_prisma_reflectance_and_wavelengths(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PRISMA reflectance_coregistered.tif and extract wavelengths from metadata.

    Returns:
        img: [C,H,W] float32, clipped to [0,1]
        wavelengths_nm: [C] float32, central wavelength per band in nm
    """
    with rasterio.open(path) as ds:
        img = ds.read().astype(np.float32)  # [C,H,W]

        # Clip to [0,1] just to be safe
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        # --- Try to read wavelengths from metadata ---
        tags_global = ds.tags()
        wavelengths_nm = None

        # 1) Global tag containing all wavelengths
        for key in ["WAVELENGTH", "wavelength", "Wavelength"]:
            if key in tags_global:
                raw = tags_global[key]
                # often comma- or space-separated
                parts = [p for p in raw.replace(";", ",").replace(" ", ",").split(",") if p]
                wavelengths_nm = np.array([float(p) for p in parts], dtype=np.float32)
                break

        # 2) If not found, try per-band tags
        if wavelengths_nm is None:
            wl_list: List[float] = []
            for bidx in range(1, ds.count + 1):
                band_tags = ds.tags(bidx=bidx)
                if "WAVELENGTH" in band_tags:
                    wl_list.append(float(band_tags["WAVELENGTH"]))
                elif "wavelength" in band_tags:
                    wl_list.append(float(band_tags["wavelength"]))
                else:
                    # First time we fail, print some info and raise
                    print(f"[WARN] No wavelength key found for band {bidx}. "
                          f"Available keys: {list(band_tags.keys())}")
                    raise RuntimeError(
                        f"Cannot find wavelength metadata for PRISMA file {path}"
                    )
            wavelengths_nm = np.array(wl_list, dtype=np.float32)

        if wavelengths_nm.shape[0] != img.shape[0]:
            raise RuntimeError(
                f"Mismatch between bands in image ({img.shape[0]}) "
                f"and wavelengths ({wavelengths_nm.shape[0]}) in {path}"
            )

    return img, wavelengths_nm


def prepare_tensor_for_hyperfree(
    img_chw: np.ndarray,
    img_size: int,
) -> torch.Tensor:
    """
    Take a [C,H,W] numpy array in [0,1], resize to [C,img_size,img_size]
    and return a [1,C,img_size,img_size] float32 torch tensor on DEVICE.
    """
    tensor = torch.from_numpy(img_chw).unsqueeze(0)  # [1,C,H,W]
    tensor = tensor.to(DEVICE)

    _, _, H, W = tensor.shape
    if H != img_size or W != img_size:
        tensor = F.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    return tensor  # [1,C,img_size,img_size]


def encode_image(
    encoder: torch.nn.Module,
    img_chw: np.ndarray,
    wavelengths_nm: np.ndarray,
    img_size: int,
    gsd_meters: float,
) -> torch.Tensor:
    """
    Run HyperFree encoder on a single image [C,H,W] with associated wavelengths and GSD.
    Returns embedding vector [D] (torch.float32 on DEVICE).
    """
    assert img_chw.ndim == 3
    assert img_chw.shape[0] == wavelengths_nm.shape[0], \
        "Number of bands must match number of wavelengths."

    tensor = prepare_tensor_for_hyperfree(img_chw, img_size)  # [1,C,H,W]

    wavelengths_list = wavelengths_nm.astype(float).tolist()
    gsd_list = [float(gsd_meters)]

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        feats = encoder(
            tensor,
            test_mode=True,
            input_wavelength=wavelengths_list,
            GSD=gsd_list,
        )
        z = get_embedding_from_feature_map(feats)  # [1,D]

    return z.squeeze(0)  # [D]


def find_pair_files(folder: Path) -> Tuple[Path, Path]:
    """
    Given a folder for a single acquisition, find:
      - sentinel_mosaic.tif
      - *reflectance_coregistered.tif
    """
    s2 = None
    prisma = None

    for tif in folder.glob("*.tif"):
        name = tif.name.lower()
        if "sentinel_mosaic" in name:
            s2 = tif
        elif "reflectance_coregistered" in name:
            prisma = tif

    if s2 is None or prisma is None:
        raise RuntimeError(
            f"Could not find both sentinel_mosaic and *reflectance_coregistered in {folder}"
        )

    return s2, prisma


def cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> float:
    z1n = z1 / (z1.norm(p=2) + 1e-8)
    z2n = z2 / (z2.norm(p=2) + 1e-8)
    return float(torch.dot(z1n, z2n).item())


# ================================
# MAIN
# ================================

def main():
    encoder, img_size = get_hyperfree_encoder()
    print(f"[INFO] Using HyperFree encoder with img_size={img_size}, device={DEVICE}")

    if not ROOT_DIR.is_dir():
        raise RuntimeError(f"Root directory {ROOT_DIR} does not exist.")

    acquisition_folders = sorted(
        [p for p in ROOT_DIR.iterdir() if p.is_dir()]
    )

    if not acquisition_folders:
        raise RuntimeError(f"No acquisition folders found under {ROOT_DIR}")

    print(f"[INFO] Found {len(acquisition_folders)} acquisition folders.")

    for folder in acquisition_folders:
        print(f"\n=== Acquisition: {folder.name} ===")

        s2_path, prisma_path = find_pair_files(folder)
        print(f"[INFO] Sentinel-2 file: {s2_path.name}")
        print(f"[INFO] PRISMA file:      {prisma_path.name}")

        # ---- Load Sentinel-2 ----
        s2_img = load_sentinel2_reflectance(s2_path)  # [C,H,W]
        s2_wl = S2_WAVELENGTHS_NM

        if s2_img.shape[0] != len(s2_wl):
            raise RuntimeError(
                f"Sentinel-2 band count ({s2_img.shape[0]}) does not match "
                f"wavelengths ({len(s2_wl)})"
            )

        # ---- Load PRISMA ----
        prisma_img, prisma_wl = load_prisma_reflectance_and_wavelengths(prisma_path)
        print(f"[INFO] PRISMA bands: {prisma_img.shape[0]}")

        # ---- Encode both with HyperFree ----
        z_s2 = encode_image(
            encoder=encoder,
            img_chw=s2_img,
            wavelengths_nm=s2_wl,
            img_size=img_size,
            gsd_meters=GSD_S2,
        )

        z_prisma = encode_image(
            encoder=encoder,
            img_chw=prisma_img,
            wavelengths_nm=prisma_wl,
            img_size=img_size,
            gsd_meters=GSD_PRISMA,
        )

        # ---- Compute distances ----
        euclidean_dist = float(torch.norm(z_s2 - z_prisma, p=2).item())
        cos_sim = cosine_similarity(z_s2, z_prisma)

        print(f"[RESULT] Embedding L2 distance (PRISMA vs S2): {euclidean_dist:.4f}")
        print(f"[RESULT] Cosine similarity   (PRISMA vs S2): {cos_sim:.4f}")


if __name__ == "__main__":
    main()