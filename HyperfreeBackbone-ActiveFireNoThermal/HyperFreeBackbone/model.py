"""Wrapper around HyperFree's ImageEncoderViT for standalone feature extraction."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from config import BackboneConfig
from utils.image_encoder import ImageEncoderViT


def _prepare_state_dict(raw_checkpoint: Dict[str, torch.Tensor], encoder: ImageEncoderViT) -> Dict[str, torch.Tensor]:
    """Resize positional parameters when the requested image size differs from training."""

    # NOTE: Mirrors HyperFree/build_HyperFree.load_and_resize_params but scoped to the
    # backbone, so we only mutate the tensors that the encoder actually owns.

    model_dict = encoder.state_dict()
    updated_state: Dict[str, torch.Tensor] = {}

    for key, value in raw_checkpoint.items():
        if key not in model_dict:
            # Skip segmentation head, prompt encoder, decoder, etc.
            continue

        target = model_dict[key]
        if value.shape != target.shape:
            if "pos_embed" in key:
                # Stored as [B, H, W, C]; adapt to current token grid.
                value = F.interpolate(
                    value.permute(0, 3, 1, 2),
                    size=(target.shape[1], target.shape[2]),
                    mode="nearest",
                ).permute(0, 2, 3, 1)
            elif "rel_pos" in key:
                value = F.interpolate(
                    value.unsqueeze(0).unsqueeze(0),
                    size=(target.shape[0], target.shape[1]),
                    mode="nearest",
                ).squeeze(0).squeeze(0)
            elif "weight_bank" in key:
                value = F.interpolate(
                    value,
                    size=(target.shape[2], target.shape[3]),
                    mode="nearest",
                )
        updated_state[key] = value

    model_dict.update(updated_state)
    return model_dict


def build_image_encoder(config: BackboneConfig) -> ImageEncoderViT:
    """Instantiate ImageEncoderViT with the vit-b hyperparameters used for HyperFree."""

    encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=config.img_size,
        mlp_ratio=4.0,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=config.vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=(2, 5, 8, 11),
        merge_indexs=[3, 6, 8, 11],
        window_size=14,
        out_chans=256,
        in_chans=8,
    )
    return encoder


def load_weights(encoder: ImageEncoderViT, checkpoint_path: Path) -> ImageEncoderViT:
    """Load pretrained or finetuned HyperFree weights while dropping unused components."""

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Unwrap common wrappers
    if isinstance(raw_checkpoint, dict):
        if "model" in raw_checkpoint:
            raw_checkpoint = raw_checkpoint["model"]
        elif "state_dict" in raw_checkpoint:
            raw_checkpoint = raw_checkpoint["state_dict"]
        elif "backbone_state_dict" in raw_checkpoint:
            # This is the format saved by the finetuning script
            raw_checkpoint = raw_checkpoint["backbone_state_dict"]

    if not hasattr(raw_checkpoint, "items"):
        raise TypeError(
            f"Unsupported checkpoint format at {checkpoint_path}: "
            "expected a (ordered) dict with weight tensors."
        )

    encoder_prefix = "image_encoder."
    keys = list(raw_checkpoint.keys())

    # Case 1: keys are prefixed with "image_encoder."
    if any(k.startswith(encoder_prefix) for k in keys):
        filtered_state = {
            key[len(encoder_prefix) :]: value
            for key, value in raw_checkpoint.items()
            if key.startswith(encoder_prefix)
        }
    else:
        # Case 2: raw_checkpoint already contains bare encoder keys
        filtered_state = dict(raw_checkpoint)

    if not filtered_state:
        raise KeyError(
            "No encoder weights found in checkpoint. "
            "Expected either 'image_encoder.*' keys or a 'backbone_state_dict' with encoder weights."
        )

    state_dict = _prepare_state_dict(filtered_state, encoder)
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()
    return encoder


__all__ = ["build_image_encoder", "load_weights"]
