from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

# Single source of truth for spectral metadata (nm)
SPECTRAL_DATABASE = {
    "activefire": {
        "wavelengths": [443, 482.5, 562.5, 655, 865, 1610, 2200, 1375],
        "bandwidths": [16, 60, 57, 27, 28, 95, 187, 21],
    },
    "spectralearth": {
        # Placeholder broad coverage (13 bands similar to Sentinel-2)
        "wavelengths": [443, 492, 560, 665, 704, 740, 783, 832, 864, 945, 1375, 1610, 2190],
        "bandwidths": [21, 66, 36, 31, 15, 15, 20, 106, 21, 20, 30, 94, 180],
    },
    "spectralearth7bands": {
        "wavelengths": [443, 482.5, 562.5, 655, 865, 1610, 2200],
        "bandwidths": [16, 60, 57, 27, 28, 95, 187],
    },
    "copernicuspretrain": {
        # Sentinel-2 TOA mix (13 bands)
        "wavelengths": [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190],
        "bandwidths": [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180],
    },
    "flogapre": {
        "wavelengths": [490, 560, 665, 705, 740, 783, 1610, 2190, 865],
        "bandwidths": [65, 35, 30, 15, 15, 20, 90, 180, 20],
    },
    "flogapost": {
        "wavelengths": [490, 560, 665, 705, 740, 783, 1610, 2190, 865],
        "bandwidths": [65, 35, 30, 15, 15, 20, 90, 180, 20],
    },
    "floga": {
        "wavelengths": [490, 560, 665, 705, 740, 783, 1610, 2190, 865],
        "bandwidths": [65, 35, 30, 15, 15, 20, 90, 180, 20],
    },
}


def get_spectral_metadata(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns dataset spectral metadata as numpy float32 arrays.
    """
    key = dataset_name.lower()
    if key not in SPECTRAL_DATABASE:
        raise ValueError(f"Unknown dataset for spectral metadata: {dataset_name}")
    entry = SPECTRAL_DATABASE[key]
    waves = np.asarray(entry["wavelengths"], dtype=np.float32)
    bands = np.asarray(entry["bandwidths"], dtype=np.float32)
    if waves.shape[0] != bands.shape[0]:
        raise ValueError(f"Wavelengths/Bandwidths length mismatch for dataset {dataset_name}")
    return waves, bands


# Backward-compatible helper for existing dataloader import
def resolve_wave_bandwidth(
    dataset_id: str,
    channels: int,
    wavelengths_path=None,
    bandwidth_path=None,
):
    waves_np, bands_np = get_spectral_metadata(dataset_id)
    if waves_np.shape[0] != channels:
        waves_np = np.linspace(400, 1000, channels, dtype=np.float32)
    if bands_np.shape[0] != channels:
        bands_np = np.full((channels,), 10.0, dtype=np.float32)
    wave_t = torch.tensor(waves_np, dtype=torch.float32)
    band_t = torch.tensor(bands_np, dtype=torch.float32)
    return wave_t, band_t
