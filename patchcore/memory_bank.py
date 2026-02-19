"""
PatchCore memory bank implementation.

This module provides a lightweight, self-contained version of the PatchCore
memory bank that can be trained on pre-extracted feature tensors and later used
to score test images.  The code intentionally avoids any dependency on the
original ``patchcore-inspection`` repository so that it can operate as a
standalone building block for anomaly detection experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

import utils

from tqdm import tqdm
import time


@dataclass
class MemoryBankState:
    """
    Serializable snapshot of the memory bank.

    Attributes:
        memory (torch.Tensor):
            Tensor of shape ``[num_entries, feature_dim]`` containing the patch
            descriptors chosen during coreset construction.

        feature_dim (int):
            Dimensionality of the stored feature vectors (i.e., number of
            channels coming from the encoder).

        sampling_ratio (float):
            Fraction of the original patch population that was retained during
            coreset construction.  The value is purely informative and does not
            affect inference when reloading the memory bank.

        coreset_method (str):
            Name of the strategy used to build the memory bank (e.g., ``"random"``
            or ``"greedy"``).  Stored for reproducibility and to automatically
            configure the memory bank when reloading from disk.
    """

    memory: torch.Tensor
    feature_dim: int
    sampling_ratio: float
    coreset_method: str


class MemoryBank:
    """
    Memory bank for PatchCore-style anomaly detection.

    The memory bank stores a subset of patch descriptors extracted from
    normal (i.e., non-defective) training images.  During inference, test
    descriptors are compared against this database to compute nearest-neighbor
    distances that act as anomaly scores.

    Key design choices in this simplified implementation:

    * **Coreset construction**: A random subsampling strategy is provided as a
      baseline.  While the original paper uses a greedy coreset algorithm, the
      random approach is often sufficient for prototyping and is dramatically
      simpler to implement.  The interface is designed so that a more advanced
      sampler can be plugged in later without changing the public API.

    * **Distance metric**: Euclidean (L2) distance is used to measure similarity
      between patch descriptors.  This matches the default configuration in many
      PatchCore deployments.
    """

    def __init__(
        self,
        *,
        sampling_ratio: float = 1.0,
        device: Optional[torch.device | str] = None,
        random_seed: Optional[int] = None,
        coreset_method: str = "random",
    ) -> None:
        """
        Initialize an empty memory bank.

        Args:
            sampling_ratio (float, optional):
                Fraction of all available patches to retain when constructing the
                memory bank.  ``1.0`` keeps every patch, while values in ``(0, 1]``
                perform random subsampling.  The ratio is applied *after* all
                patches from the provided feature tensors have been aggregated.

            device (Optional[Union[torch.device, str]]):
                Device where the memory bank is stored.  Defaults to CPU when
                ``None``.  Keeping the memory on CPU is often convenient because
                k-nearest-neighbor inference is typically bandwidth-bound rather
                than compute-bound, and it avoids exhausting GPU memory.

            random_seed (Optional[int]):
                Seed controlling the deterministic random subsampling step.  When
                ``None`` (default), PyTorch's global RNG state is used.

            coreset_method (str, optional):
                Name of the strategy that should be used to down-sample the patch
                population.  Supported values are ``"random"`` and ``"greedy"``.
                The greedy k-center algorithm provides better coverage at the cost
                of higher compute and memory usage.
        """

        if not (0.0 < sampling_ratio <= 1.0):
            raise ValueError("sampling_ratio must be within (0, 1].")

        self.sampling_ratio = float(sampling_ratio)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.coreset_method = coreset_method.lower()

        if self.coreset_method not in {"random", "greedy"}:
            raise ValueError(
                "coreset_method must be one of {'random', 'greedy'}, "
                f"received '{coreset_method}'."
            )

        # Internal storage for the memory tensor and its dimensionality.
        self._memory: Optional[torch.Tensor] = None
        self._feature_dim: Optional[int] = None

        # Torch generator used for reproducible subsampling when desired.
        self._rng: Optional[torch.Generator] = None
        if random_seed is not None:
            self._rng = torch.Generator(device="cpu")
            self._rng.manual_seed(int(random_seed))

    # --------------------------------------------------------------------- #
    # Helper properties
    # --------------------------------------------------------------------- #
    @property
    def is_fitted(self) -> bool:
        """Return ``True`` once the memory bank has been constructed."""

        return self._memory is not None

    @property
    def memory(self) -> torch.Tensor:
        """
        Accessor for the stored patch descriptors.

        Returns:
            torch.Tensor:
                Tensor with shape ``[num_entries, feature_dim]`` holding the
                subset of patches selected during :meth:`build`.

        Raises:
            RuntimeError: When the memory bank has not been constructed yet.
        """

        if self._memory is None:
            raise RuntimeError("Memory bank has not been built yet.")
        return self._memory

    @property
    def feature_dim(self) -> int:
        """
        Dimensionality of the patch descriptors stored in the memory bank.

        Returns:
            int: Number of channels in each patch vector.

        Raises:
            RuntimeError: When the memory bank has not been constructed yet.
        """

        if self._feature_dim is None:
            raise RuntimeError("Memory bank has not been built yet.")
        return self._feature_dim

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def build(self, feature_sources: Sequence[torch.Tensor | utils.FeatureSet]) -> None:
        """
        Construct the memory bank from a collection of feature tensors.

        Args:
            feature_sources (Sequence[Union[torch.Tensor, utils.FeatureSet]]):
                Iterable containing feature representations for one or more
                images.  Each element can be either:

                * A 3D tensor with shape ``[C, H, W]`` representing the feature
                  map for a single image.
                * A 2D tensor with shape ``[num_patches, C]`` containing already
                  flattened patch descriptors.
                * A :class:`utils.FeatureSet` instance produced by
                  :func:`utils.build_feature_sets`.

                Mixing tensor types is allowed.  All items must share the same
                feature dimensionality ``C``.

        Returns:
            None.  The constructed memory bank is stored internally and can be
            accessed via :attr:`memory`.

        Raises:
            ValueError: If input tensors have inconsistent channel counts or if
            the sequence is empty.
        """

        if not feature_sources:
            raise ValueError("Cannot build memory bank from an empty collection.")

        patch_matrices: List[torch.Tensor] = []
        feature_dim: Optional[int] = None

        for entry in feature_sources:
            if isinstance(entry, utils.FeatureSet):
                patches = entry.patches
            elif isinstance(entry, torch.Tensor):
                if entry.dim() == 3:
                    patches = utils.flatten_feature_map(entry)
                elif entry.dim() == 2:
                    patches = entry
                else:
                    raise ValueError(
                        "Feature tensors must have 2 or 3 dimensions. "
                        f"Received tensor with shape {entry.shape}."
                    )
            else:
                raise TypeError(
                    "feature_sources must contain torch.Tensor or utils.FeatureSet "
                    f"instances, got type {type(entry)}."
                )

            if patches.dim() != 2:
                raise ValueError(
                    "Flattened patch matrices must have shape [num_patches, C]."
                )

            current_dim = patches.size(1)
            if feature_dim is None:
                feature_dim = current_dim
            elif current_dim != feature_dim:
                raise ValueError(
                    "All feature tensors must share the same feature dimensionality."
                )

            patch_matrices.append(patches)

        # Concatenate along the patch dimension to collect all normal patches.
        full_patch_matrix = torch.cat(patch_matrices, dim=0).contiguous()

        # Subsample patches according to the chosen coreset strategy.
        sampled_matrix = self._select_coreset(full_patch_matrix).contiguous()

        # Move to the requested device for inference.
        self._memory = sampled_matrix.to(self.device).contiguous()
        self._feature_dim = int(feature_dim)

        # Faiss backend removed; PyTorch nearest-neighbor search is always used.


    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #
    def save(self, path: Path | str) -> None:
        """
        Serialize the memory bank to disk.

        Args:
            path (Union[pathlib.Path, str]):
                Destination file.  A ``.pt`` extension is conventional but not
                enforced.  Parent directories are created automatically.

        Returns:
            None.  The state is written using :func:`torch.save`.

        Raises:
            RuntimeError: If the memory bank has not been built yet.
        """

        if not self.is_fitted:
            raise RuntimeError("Cannot save an uninitialized memory bank.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        state = MemoryBankState(
            memory=self.memory.cpu(),  # Persist on CPU for portability.
            feature_dim=self.feature_dim,
            sampling_ratio=self.sampling_ratio,
            coreset_method=self.coreset_method,
        )

        torch.save(
            {
                "state": state,
                "metadata": {
                    "device": str(self.device),
                    "rng_seed": self._rng.initial_seed() if self._rng is not None else None,
                },
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: Optional[torch.device | str] = None,
    ) -> "MemoryBank":
        """
        Restore a memory bank from disk.

        Args:
            path (Union[pathlib.Path, str]):
                Path to the serialized memory bank produced by :meth:`save`.

            device (Optional[Union[torch.device, str]]):
                Device to host the loaded memory tensor.  Defaults to the device
                recorded at save time.  Override when you want to move the memory
                bank to a different device (e.g., CPU to GPU).

        Returns:
            MemoryBank: A fully constructed memory bank ready for inference.

        Raises:
            FileNotFoundError: If the provided path does not exist.
            KeyError: If the serialized file is missing expected fields.
        """

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Memory bank file not found: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        try:
            state: MemoryBankState = payload["state"]
            metadata = payload["metadata"]
        except KeyError as exc:
            raise KeyError("Serialized memory bank is missing required keys.") from exc

        restore_device = (
            torch.device(device) if device is not None else torch.device(metadata["device"])
        )

        rng_seed = metadata.get("rng_seed")
        mem_bank = cls(
            sampling_ratio=state.sampling_ratio,
            device=restore_device,
            random_seed=rng_seed,
            coreset_method=state.coreset_method,
        )

        mem_bank._memory = state.memory.to(restore_device)
        mem_bank._feature_dim = state.feature_dim

        return mem_bank

    # --------------------------------------------------------------------- #
    # Inference
    # --------------------------------------------------------------------- #
    def compute_anomaly_scores(
        self,
        feature_tensors: Sequence[torch.Tensor | utils.FeatureSet],
        *,
        reduction: str = "mean",
        chunk_size: int = 1024,
    ) -> List[Tuple[str, float]]:
        """
        Compute image-level anomaly scores using 1-NN distances.

        Args:
            feature_tensors (Sequence[Union[torch.Tensor, utils.FeatureSet]]):
                Iterable holding feature maps for the test images.  The accepted
                formats mirror those of :meth:`build`.  When a tensor or
                ``FeatureSet`` lacks an explicit image name, a placeholder name is
                generated for reporting purposes.

            reduction (str, optional):
                Strategy used to aggregate patch-level distances into a single
                image-level score.  Currently supported options:
                    * ``"mean"`` (default): average distance across patches.
                    * ``"max"``: maximum distance across patches.
                Additional reductions can be added later if needed.

            chunk_size (int, optional):
                Number of patch vectors to process simultaneously when computing
                pairwise distances.  Chunking prevents excessive memory usage
                when images contain many patches.  Values that are multiples of
                the number of memory entries generally yield good performance.

        Returns:
            List[Tuple[str, float]]:
                List of ``(image_name, anomaly_score)`` pairs, preserving the
                input ordering.  Each score quantifies how dissimilar the image
                is from the training distribution; higher values indicate higher
                anomaly likelihood.

            Dictionary[str, List[float]]: A dictionary mapping image names to their
            patch-level anomaly scores.

        Raises:
            RuntimeError: If the memory bank was not built prior to calling this
            method.
            ValueError: For unsupported reduction strategies.
        """

        if not self.is_fitted:
            raise RuntimeError("Memory bank must be built before scoring anomalies.")

        if reduction not in ("mean", "max"):
            raise ValueError(f"Unsupported reduction strategy: {reduction}")

        image_scores: List[Tuple[str, float]] = []
        patch_scores: dict[str, List[float]] = {}

        for index, entry in tqdm(enumerate(feature_tensors), total=len(feature_tensors), desc="Scoring images"):
            if isinstance(entry, utils.FeatureSet):
                image_name = entry.image_name
                patch_matrix = entry.patches
            elif isinstance(entry, torch.Tensor):
                if entry.dim() == 3:
                    image_name = f"image_{index:05d}.pt"
                    patch_matrix = utils.flatten_feature_map(entry)
                elif entry.dim() == 2:
                    image_name = f"image_{index:05d}.pt"
                    patch_matrix = entry
                else:
                    raise ValueError(
                        "Feature tensors must have 2 or 3 dimensions for scoring."
                    )
            else:
                raise TypeError(
                    "feature_tensors must contain torch.Tensor or utils.FeatureSet "
                    f"instances, got type {type(entry)}."
                )

            if patch_matrix.size(1) != self.feature_dim:
                raise ValueError(
                    "Feature dimensionality mismatch during scoring. "
                    f"Expected {self.feature_dim}, got {patch_matrix.size(1)}."
                )

            distances = self._compute_patch_distances(patch_matrix, chunk_size=chunk_size)
            patch_scores[image_name] = distances.cpu().tolist()

            if reduction == "mean":
                score = distances.mean().item()
            else:  # reduction == "max"
                score = distances.max().item()

            image_scores.append((image_name, score))

        return image_scores, patch_scores

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _compute_patch_distances(
        self,
        patch_matrix: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Compute distances between a set of patches and the memory bank.

        Args:
            patch_matrix (torch.Tensor):
                Tensor with shape ``[num_patches, feature_dim]`` representing all
                patches extracted from a single image.

            chunk_size (int):
                Maximum number of patches to process simultaneously.  Keeping the
                chunk size modest (e.g., 1024) prevents the pairwise distance
                matrix from consuming excessive memory when the memory bank is
                large.

        Returns:
            torch.Tensor:
                1D tensor of shape ``[num_patches]`` where each element is the
                distance to the nearest memory bank entry.
        """

        device = self.memory.device
        memory = self.memory

        # Move patches to the same device as the memory bank for efficient matmul.
        patches = patch_matrix.to(device)

        num_patches = patches.size(0)
        nearest_distances: List[torch.Tensor] = []

        # Iterate over chunks so that we only compute distance matrices of size
        # [chunk_size, num_memory_entries] at a time.
        for start in range(0, num_patches, chunk_size):
            end = min(start + chunk_size, num_patches)
            patch_chunk = patches[start:end]

            # Pairwise distances between the chunk and the entire memory bank.
            dists = utils.compute_pairwise_l2(patch_chunk, memory, squared=False)

            # Keep the minimum distance for each patch in the chunk.
            nearest_distances.append(dists.min(dim=1).values)

        # Concatenate chunk-level minima to recover the full list of closest distances.
        return torch.cat(nearest_distances, dim=0)

    # ------------------------------------------------------------------ #
    # Coreset selection strategies
    # ------------------------------------------------------------------ #
    def _select_coreset(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Dispatch helper that applies the configured coreset strategy.

        Args:
            patches (torch.Tensor):
                Tensor of shape ``[num_patches, feature_dim]`` with the full set
                of normal descriptors collected during :meth:`build`.

        Returns:
            torch.Tensor:
                Tensor containing the selected subset of patches according to the
                configured coreset strategy and sampling ratio.
        """

        if self.sampling_ratio >= 1.0:
            return patches

        total_patches = patches.size(0)
        sample_size = max(1, int(total_patches * self.sampling_ratio))

        if self.coreset_method == "random":
            return self._random_subsample(patches, sample_size).contiguous()

        if self.coreset_method == "greedy":
            return self._greedy_kcenter(patches, sample_size).contiguous()

        # The guard above should prevent reaching this point, but keep for safety.
        raise RuntimeError(f"Unsupported coreset method: {self.coreset_method}")

    def _random_subsample(self, patches: torch.Tensor, sample_size: int) -> torch.Tensor:
        """
        Select a random subset of patch descriptors.

        Args:
            patches (torch.Tensor):
                Full population of normal patches with shape ``[N, C]``.

            sample_size (int):
                Number of patches to retain in the memory bank.  Must satisfy
                ``1 <= sample_size <= N``.

        Returns:
            torch.Tensor:
                Subsampled tensor with shape ``[sample_size, C]``.
        """

        total_patches = patches.size(0)
        if sample_size >= total_patches:
            return patches

        generator = self._rng
        indices = torch.randperm(total_patches, generator=generator)[:sample_size]
        return patches.index_select(0, indices).contiguous()

    def _greedy_kcenter(self, patches: torch.Tensor, sample_size: int) -> torch.Tensor:
        """
        Apply the greedy k-center algorithm described in the PatchCore paper.

        Args:
            patches (torch.Tensor):
                Tensor with shape ``[N, C]`` representing all available patch
                descriptors extracted from normal images.

            sample_size (int):
                Target number of descriptors to keep.  Must satisfy
                ``1 <= sample_size <= N``.

        Returns:
            torch.Tensor:
                Tensor of shape ``[sample_size, C]`` containing the selected patch
                descriptors that best cover the feature distribution by maximizing
                the minimum distance to already selected centers.

        Implementation details:
            * The algorithm begins with a randomly chosen seed patch, then
              iteratively adds the patch that is farthest from the current set of
              selected centers.
            * To avoid recomputing distances to all selected centers, we maintain
              a running vector of the minimum squared distances from each patch
              to the set of chosen centers.
            * Squared distances are used for efficiency since they preserve the
              ordering induced by Euclidean distance.
        """

        num_patches = patches.size(0)
        if sample_size >= num_patches:
            return patches

        # Ensure sampling happens on CPU for deterministic behavior even if the
        # patches currently reside on a GPU.
        generator = self._rng
        initial_index = torch.randint(
            high=num_patches,
            size=(1,),
            generator=generator,
        ).item()

        selected_indices = [initial_index]

        # Initialize the distance tracker with distances to the first center.
        current_center = patches[initial_index : initial_index + 1]
        min_sq_distances = self._squared_l2_to_center(patches, current_center)

        tqdm_iter = tqdm(total=sample_size - 1, desc="Building memory bank", unit="patch")

        while len(selected_indices) < sample_size:
            # Identify the patch farthest from the existing centers.
            next_index = torch.argmax(min_sq_distances).item()
            selected_indices.append(next_index)

            # Update minimum distances using the newly added center.
            new_center = patches[next_index : next_index + 1]
            new_sq_distances = self._squared_l2_to_center(patches, new_center)
            min_sq_distances = torch.minimum(min_sq_distances, new_sq_distances)

            tqdm_iter.update(1)

        tqdm_iter.close()

        index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=patches.device)
        return patches.index_select(0, index_tensor).contiguous()

    @staticmethod
    def _squared_l2_to_center(patches: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """
        Compute squared L2 distances between every patch and a single center.

        Args:
            patches (torch.Tensor):
                Tensor with shape ``[N, C]`` containing the full patch population.

            center (torch.Tensor):
                Tensor with shape ``[1, C]`` representing the center vector.

        Returns:
            torch.Tensor:
                1D tensor of shape ``[N]`` containing squared Euclidean distances.
        """

        diff = patches - center
        return diff.pow(2).sum(dim=1)
