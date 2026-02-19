from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functorch import vmap
from torch import Tensor

from .utils import to_2tuple


def pi_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via pseudo-inverse resizing."""
    assert len(patch_embed.shape) == 4
    assert len(new_patch_size) == 2

    old_patch_size = tuple(patch_embed.shape[2:])
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size).to(
        patch_embed.device
    )

    def resample_patch_embed(patch_embed: Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)
    return v_resample_patch_embed(patch_embed)


def interpolate_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via interpolation."""
    assert len(patch_embed.shape) == 4
    assert len(new_patch_size) == 2

    return F.interpolate(
        patch_embed, new_patch_size, mode=interpolation, antialias=antialias
    )


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 240,
        patch_size: Union[int, Tuple[int, int]] = 32,
        grid_size: Union[int, Tuple[int, int]] = 7,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
        patch_size_probs: Optional[Sequence[float]] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = to_2tuple(grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.interpolation = interpolation
        self.antialias = antialias

        self.patch_size_seq = patch_size_seq
        if self.patch_size_seq:
            if not patch_size_probs:
                n = len(self.patch_size_seq)
                self.patch_size_probs = [1.0 / n] * n
            else:
                self.patch_size_probs = [
                    p / sum(patch_size_probs) for p in patch_size_probs
                ]
        else:
            self.patch_size_probs = []

        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        pinvs = {}
        for ps in self.patch_size_seq:
            ps = to_2tuple(ps)
            pinvs[ps] = self._calculate_pinv(self.patch_size, ps)
        return pinvs

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, old_shape: Tuple[int, int], new_shape: Tuple[int, int]
    ) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def _resample_patch_embed(self, patch_embed: Tensor, patch_size: Tuple[int, int]):
        h, w = patch_size
        resampled_kernel = self.pinvs[patch_size] @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    def forward(self, x: Tensor, patch_size: Optional[Tuple[int, int]] = None):
        patch_size = patch_size or self.patch_size
        patch_size = to_2tuple(patch_size)
        if patch_size != self.patch_size:
            new_proj_weight = torch.empty(
                self.proj.weight.shape[0],
                self.proj.weight.shape[1],
                patch_size[0],
                patch_size[1],
                device=x.device,
                dtype=x.dtype,
            )
            for i in range(self.proj.weight.shape[0]):
                for j in range(self.proj.weight.shape[1]):
                    new_proj_weight[i, j] = self._resample_patch_embed(
                        self.proj.weight[i, j], patch_size
                    )
            x = F.conv2d(x, new_proj_weight, self.proj.bias, patch_size, patch_size)
        else:
            x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)
        return x
