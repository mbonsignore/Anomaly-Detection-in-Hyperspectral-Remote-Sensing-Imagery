# Copernicus-FM Extracted Backbone

Self-contained feature extractor built from the Copernicus-FM repository, structured like the HyperFree backbone for drop-in use in PatchCore-style pipelines.

## Files
- `config.py` — dataset defaults, spectral metadata paths, grid/pooling options.
- `dataloader.py` — loads patches (.tif or .pt), normalizes per channel, attaches wavelength/bandwidth and meta_info tensors.
- `model.py` — wrapper around `CopernicusFMViT` with optional 4x4 pooling, deterministic setup, weight loading from `ckpt/`.
- `main.py` — CLI for batch feature extraction to `.pt` feature maps.
- `utils/` — minimal model definition and dynamic hypernetwork utilities (no training code).
- `ckpt/CopernicusFM_ViT_base_varlang_e100.pth` — pretrained ViT-B/16 weights.

## Basic usage
```bash
python main.py --dataset activefire --data-dir /path/to/tiles \
  --out-dir ./features_cfm --device cuda:0 --output-grid 4x4
```

Arguments of interest:
- `--output-grid`: `4x4` (default) applies adaptive avg pooling from the native grid (e.g., 16x16) to 4x4; use `native` to keep the original resolution.
- `--kernel-size`: patch-embed kernel/stride; keep at `16` for the provided checkpoint.
- `--feature-block-index`: transformer block to tap for patch tokens (default: last block for ViT-B/16 → 11).
- `--wavelengths-path` / `--bandwidth-path`: optional `.npy` files with per-channel spectral metadata; if absent, safe defaults are generated.

## Inputs expected by the model
- `image`: float tensor `[B, C, H, W]` (channels must align with `wave_list`/`bandwidth`).
- `meta_info`: `[B, 4]` = [lon, lat, time, area]; use NaNs if unavailable (the model handles NaNs with learned tokens).
- `wave_list` / `bandwidth`: `[C]` tensors in nanometers; generated in `dataloader.py` via `utils/spectral_meta.py` (per-dataset defaults or evenly spaced fallback).

## Outputs and PatchCore compatibility
- Saved features are per-sample tensors shaped:
  - Native mode: `[C_feat, H_feat, W_feat]` (e.g., `[768, 16, 16]` for 224x224 inputs).
  - `4x4` mode: `[C_feat, 4, 4]` via adaptive average pooling for drop-in compatibility with HyperFree’s 4x4 features.
- Features are the final block patch-token map (cls token discarded).

## Extending spectral metadata
Edit `utils/spectral_meta.py` or point `--wavelengths-path/--bandwidth-path` to `.npy` files. If lengths don’t match the image channel count, the loader will fallback to evenly spaced wavelengths and unit bandwidths with a warning (printed once).
