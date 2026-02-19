import math

import numpy as np
import torch

radius_earth = 6378137.0


def area(quad: torch.Tensor) -> torch.Tensor:
    """Area of spherical quadrilateral defined by latitude/longitude corners."""
    assert quad.shape[-2:] == (4, 2)
    lat1, lon1 = torch.deg2rad(quad[..., 0, 0]), torch.deg2rad(quad[..., 0, 1])
    lat2, lon2 = torch.deg2rad(quad[..., 1, 0]), torch.deg2rad(quad[..., 1, 1])
    lat3, lon3 = torch.deg2rad(quad[..., 2, 0]), torch.deg2rad(quad[..., 2, 1])
    lat4, lon4 = torch.deg2rad(quad[..., 3, 0]), torch.deg2rad(quad[..., 3, 1])

    def spherical_excess(lat_a, lon_a, lat_b, lon_b, lat_c, lon_c):
        a = central_angle(lat_a, lon_a, lat_b, lon_b)
        b = central_angle(lat_b, lon_b, lat_c, lon_c)
        c = central_angle(lat_c, lon_c, lat_a, lon_a)
        s = (a + b + c) / 2
        return (
            4 * torch.atan(torch.sqrt(torch.tan(s / 2) * torch.tan((s - a) / 2)
                                      * torch.tan((s - b) / 2) * torch.tan((s - c) / 2)))
        )

    def central_angle(lat_a, lon_a, lat_b, lon_b):
        return torch.acos(
            torch.sin(lat_a) * torch.sin(lat_b)
            + torch.cos(lat_a) * torch.cos(lat_b) * torch.cos(lon_a - lon_b)
        )

    e1 = spherical_excess(lat1, lon1, lat2, lon2, lat3, lon3)
    e2 = spherical_excess(lat1, lon1, lat3, lon3, lat4, lon4)
    area_sr = (e1 + e2) * radius_earth * radius_earth
    return area_sr
