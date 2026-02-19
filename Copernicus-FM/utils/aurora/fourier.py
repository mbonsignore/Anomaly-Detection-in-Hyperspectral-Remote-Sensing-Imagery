import math

import numpy as np
import torch
import torch.nn as nn

from .area import area, radius_earth

__all__ = [
    "FourierExpansion",
    "pos_expansion",
    "scale_expansion",
    "lead_time_expansion",
    "levels_expansion",
    "absolute_time_expansion",
]


class FourierExpansion(nn.Module):
    """Fourier series-style expansion into a high-dimensional space."""

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        in_range = torch.logical_and(self.lower <= x.abs(), torch.all(x.abs() <= self.upper))
        in_range_or_zero = torch.all(torch.logical_or(in_range, x == 0))
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range [{self.lower}, {self.upper}]."
            )

        if not (d % 2 == 0):
            raise ValueError("The dimensionality must be a multiple of two.")

        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return encoding.float()


_delta = 0.01
_min_patch_area: float = area(
    torch.tensor(
        [
            [90, 0],
            [90, _delta],
            [90 - _delta, _delta],
            [90 - _delta, 0],
        ],
        dtype=torch.float64,
    )
).item()
_area_earth = 4 * np.pi * radius_earth * radius_earth

pos_expansion = FourierExpansion(_delta, 720)
scale_expansion = FourierExpansion(_min_patch_area, _area_earth)
lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)
levels_expansion = FourierExpansion(0.01, 1e5)
absolute_time_expansion = FourierExpansion(1, 24 * 365.25, assert_range=False)
spectrum_central_expansion = FourierExpansion(1e-7, 1)
spectrum_width_expansion = FourierExpansion(1e-7, 1)
