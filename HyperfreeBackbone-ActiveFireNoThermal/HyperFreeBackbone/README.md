# HyperFreeBackbone

Standalone wrapper around the official [HyperFree](https://github.com/) encoder that
extracts features for Landsat-8 ActiveFire patches without the segmentation head.

## Layout

- `config.py` - edit paths and runtime knobs (batch size, device, patch size).
- `dataloader.py` - loads TIFF patches for the selected dataset, resizes to 1024x1024, and attaches the wavelength vector.
- `model.py` - rebuilds `ImageEncoderViT` with vit-b hyperparameters and loads weights.
- `main.py` - drives feature extraction and writes one `.pt` tensor per input tile.
- `utils/` - modules copied from HyperFree (`image_encoder`, spectral utils, etc.).
- `features/` - output directory for features saved by `torch.save`.

## Usage

1. Pick the dataset by setting `dataset_type = "activefire"` or `"hyperseg"` in `config.py` (defaults to ActiveFire).
2. Ensure `ckpt/HyperFree-b.pth` exists or set `HYPERFREE_BACKBONE_CKPT` / `BackboneConfig.checkpoint_path` accordingly.
3. For ActiveFire, keep patches in `ActiveFire-NoThermalBands/zXXXXX/*.tif` and ensure masks folders remain alongside. For HyperSeg, place `.tif` files directly under `HyperSeg/`.
4. Provide the matching wavelength `.npy` files (`wavelengths_landsat8_no_thermal.npy` or `wavelengths_hyperseg.npy`) in the project root.
5. Run the extractor from the project root:

   ```bash
   cd HyperfreeBackbone-ActiveFireNoThermal
   python3 HyperFreeBackbone/main.py
   ```

Each feature file has shape `[256, 16, 16]` (channels, tokens_y, tokens_x) and aligns
with HyperFree's segmentation backbone output for vit-b.
