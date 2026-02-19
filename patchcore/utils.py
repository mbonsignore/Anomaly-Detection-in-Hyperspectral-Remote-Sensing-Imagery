"""
Utility functions for the standalone PatchCore anomaly detection pipeline.

This module intentionally recreates only the minimum functionality that is
required by the simplified PatchCore implementation in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import time
from tqdm import tqdm

import torch

def iter_patches_from_files(files, device=None):
    for p in tqdm(files, desc="[TQDM] Building memory"):
        fm = torch.load(p, map_location=device)
        patches = flatten_feature_map(fm)  # [16, C] for 4x4
        yield patches


def list_feature_files(folder: Path | str) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Feature folder does not exist: {folder_path}")
    files = sorted(folder_path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt feature files found in directory: {folder_path}")
    return files

def flatten_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3D feature map into a 2D matrix of patch descriptors.

    Args:
        feature_map (torch.Tensor):
            Tensor containing the feature map for a single image.  Expected
            shape is ``[C, H, W]`` where:
                * ``C`` is the number of channels coming from the encoder.
                * ``H`` and ``W`` are the spatial dimensions (number of patches
                  along height and width respectively).

            The tensor may reside on CPU or GPU memory; this function does not
            move it to a different device.

    Returns:
        torch.Tensor:
            2D tensor with shape ``[H * W, C]``.  Each row corresponds to the
            flattened descriptor of a single spatial patch, while columns
            represent the feature channels.  The returned tensor is a view
            whenever possible and shares storage with the input.

    Raises:
        ValueError: If ``feature_map`` is not a 3D tensor of shape ``[C, H, W]``.

    Notes:
        * Flattening is required because PatchCore stores a memory bank of patch
          descriptors.  The greedy coreset selection algorithm, as well as
          nearest-neighbor queries, operate on flattened patch vectors.
        * The ``permute`` operation is used instead of ``reshape`` alone to
          ensure that the resulting layout is ``[num_patches, channels]``.
    """

    if feature_map.dim() != 3:
        raise ValueError(
            f"Expected a 3D tensor with shape [C, H, W], got {feature_map.shape}"
        )

    channels, height, width = feature_map.shape

    # Rearrange tensor to shape [H, W, C] so that patches become contiguous rows.
    # ``contiguous`` is called in case the permute changes the underlying layout.
    flattened = feature_map.permute(1, 2, 0).contiguous().view(height * width, channels)

    return flattened


def compute_pairwise_l2(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    squared: bool = False,
) -> torch.Tensor:
    """
    Compute pairwise L2 distances between two sets of feature vectors.

    Args:
        source (torch.Tensor):
            Tensor of shape ``[N, C]`` containing ``N`` source feature vectors.
            ``C`` represents the feature dimensionality.  ``N`` can be any value
            greater than zero.

        target (torch.Tensor):
            Tensor of shape ``[M, C]`` containing ``M`` target feature vectors
            with the same feature dimensionality ``C`` as ``source``.

        squared (bool, optional):
            If ``True`` the function returns squared Euclidean distances.  When
            ``False`` (default) the exact Euclidean norm is returned.  Squared
            distances are faster to compute and may be sufficient when only the
            relative ordering of distances matters.

    Returns:
        torch.Tensor:
            Distance matrix of shape ``[N, M]`` where ``dist[i, j]`` corresponds
            to the Euclidean distance between ``source[i]`` and ``target[j]``.
            The output resides on the same device as the input tensors and uses
            the default floating-point dtype dictated by PyTorch's type
            promotion rules.

    Raises:
        ValueError: If input tensors do not have shape ``[*, C]`` with matching
        feature dimensionality.

    Implementation details:
        * To avoid explicit loops, the computation leverages the identity
          ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b^T`` which is numerically
          stable and efficient.
        * Small negative values introduced by floating point rounding errors are
          clamped to zero before optionally taking the square root.
    """

    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(
            "Both source and target must be 2D tensors of shape [N, C] and [M, C]."
        )

    if source.size(1) != target.size(1):
        raise ValueError(
            "Source and target tensors must have the same feature dimensionality."
        )

    # Compute squared norms of each vector: [N] and [M] respectively.
    source_norm = source.pow(2).sum(dim=1, keepdim=True)  # Shape: [N, 1]
    target_norm = target.pow(2).sum(dim=1, keepdim=True).t()  # Shape: [1, M]

    # Pairwise squared distances using broadcasting: [N, M].
    sq_distances = source_norm + target_norm - 2.0 * source @ target.t()

    # Numerical stability: clamp tiny negatives caused by floating point errors.
    sq_distances = torch.clamp(sq_distances, min=0.0)

    if squared:
        return sq_distances

    return torch.sqrt(sq_distances + 1e-12)


def load_feature_tensors(
    folder: Path | str,
    *,
    device: Optional[torch.device | str] = None,
    allow_empty: bool = False,
) -> List[Tuple[str, torch.Tensor]]:
    """
    Load feature tensors from a directory containing ``.pt`` files.

    Args:
        folder (Union[pathlib.Path, str]):
            Path to the directory that stores encoded feature tensors.  The
            function expects each ``.pt`` file to contain a single ``torch.Tensor``
            with shape ``[C, H, W]`` representing the feature map of an image.

        device (Optional[Union[torch.device, str]]):
            Optional device specification (e.g., ``"cpu"``, ``"cuda:0"``).  When
            provided, each loaded tensor is moved to that device.  If ``None``
            (default), tensors remain on the device they were serialized from,
            which is typically CPU.

        allow_empty (bool, optional):
            When ``False`` (default), the function raises ``FileNotFoundError``
            if the folder does not contain any ``.pt`` files.  Set to ``True`` if
            empty folders should simply return an empty list without raising.

    Returns:
        List[Tuple[str, torch.Tensor]]:
            List of ``(filename, tensor)`` pairs sorted alphabetically by file
            name.  The filename is provided without the directory prefix to make
            downstream reporting (e.g., CSV outputs) easier.

    Raises:
        FileNotFoundError: If the folder is missing or lacks ``.pt`` files while
        ``allow_empty`` is ``False``.

    Usage pattern:
        >>> feature_list = load_feature_tensors("features/normal")
        >>> first_tensor = feature_list[0][1]  # torch.Tensor with shape [C, H, W]
    """

    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Feature folder does not exist: {folder_path}")

    file_paths = sorted(folder_path.glob("*.pt"))

    if not file_paths and not allow_empty:
        raise FileNotFoundError(
            f"No .pt feature files found in directory: {folder_path}"
        )

    loaded_features: List[Tuple[str, torch.Tensor]] = []

    for file_path in file_paths:
        tensor = torch.load(file_path, map_location=device)

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"File {file_path} does not contain a torch.Tensor.")

        loaded_features.append((file_path.name, tensor))

    return loaded_features


@dataclass(frozen=True)
class FeatureSet:
    """
    Lightweight container representing a batch of flattened feature patches.

    This convenience data structure bundles together the image identifier,
    flattened feature matrix, and (optionally) the original spatial dimensions.

    Attributes:
        image_name (str):
            Base filename of the feature tensor (e.g., ``"sample_001.pt"``).

        patches (torch.Tensor):
            Tensor with shape ``[num_patches, feature_dim]`` created by applying
            ``flatten_feature_map`` to the original feature map.

        spatial_size (Tuple[int, int]):
            Tuple ``(H, W)`` storing the original spatial resolution of the
            feature map.  The information is useful for future extensions such as
            reconstructing patch-level anomaly maps.
    """

    image_name: str
    patches: torch.Tensor
    spatial_size: Tuple[int, int]


def build_feature_sets(
    feature_pairs: Sequence[Tuple[str, torch.Tensor]],
) -> List[FeatureSet]:
    """
    Transform a list of raw feature tensors into flattened ``FeatureSet`` items.

    Args:
        feature_pairs (Sequence[Tuple[str, torch.Tensor]]):
            Iterable of ``(filename, tensor)`` pairs as produced by
            :func:`load_feature_tensors`.  Each tensor must have shape ``[C, H, W]``.

    Returns:
        List[FeatureSet]:
            List of ``FeatureSet`` instances containing flattened patch matrices
            ready for coreset selection or distance computations.  The ordering
            matches the input order.
    """

    feature_sets: List[FeatureSet] = []
    for filename, tensor in feature_pairs:
        patches = flatten_feature_map(tensor)
        spatial_size = (tensor.size(1), tensor.size(2))
        feature_sets.append(FeatureSet(filename, patches, spatial_size))

    return feature_sets


