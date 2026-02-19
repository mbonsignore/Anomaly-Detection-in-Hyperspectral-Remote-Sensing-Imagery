"""Wrapper around Copernicus-FM ViT for standalone feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import BackboneConfig
from utils.model_vit import vit_base_patch16


def set_deterministic() -> None:
    torch.manual_seed(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CopernicusBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        block_index = config.feature_block_index if config.feature_block_index is not None else 11
        self.model = vit_base_patch16(
            img_size=config.img_size,
            num_classes=0,
            global_pool=True,
            return_intermediate=True,
            intermediate_indices=[block_index],
        )
        # Note: vit_base_patch16 fixes patch_size=16 to match the pretrained checkpoint.
        # Runtime stride/kernel can still be adjusted via self.config.kernel_size in forward().

        self.load_weights(config.checkpoint_path)
        self.eval()

        self.pool = None
        if config.output_grid_size.lower() == "4x4":
            self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def load_weights(self, checkpoint_path) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        msg = self.model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}: {msg}")

    def forward(
        self,
        images: torch.Tensor,
        meta_info: torch.Tensor,
        wave_list: torch.Tensor,
        bandwidth: torch.Tensor,
        language_embed=None,
    ) -> torch.Tensor:
        # Ensure spectral metadata is 1D [C] (hypernetwork expects per-band vectors, not batched)
        if wave_list.dim() > 1:
            wave_list = wave_list[0]
        if bandwidth.dim() > 1:
            bandwidth = bandwidth[0]

        _, feats = self.model(
            images,
            meta_info,
            wave_list,
            bandwidth,
            language_embed,
            self.config.input_mode,
            self.config.kernel_size,
        )
        feature_map = feats[-1]
        if self.pool is not None:
            feature_map = self.pool(feature_map)
        return feature_map


def build_image_encoder(config: BackboneConfig) -> CopernicusBackbone:
    set_deterministic()
    model = CopernicusBackbone(config)
    return model


__all__ = ["CopernicusBackbone", "build_image_encoder"]
