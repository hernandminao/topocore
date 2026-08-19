"""
topocore.analysis.quality.c2c
==============================

Cloud-to-Cloud (C2C) distance analysis.

Computes nearest-neighbor distances between two point clouds.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import QualityError
from topocore.analysis.types import CloudToCloudResult

try:
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True

except ImportError:
    _HAS_SCIPY = False


class CloudToCloudDistance:
    """
    Computes Cloud-to-Cloud distances.

    Parameters
    ----------
    max_distance
        Maximum search distance.
        Zero means unlimited.
    """

    __slots__ = ("_max_distance",)

    def __init__(
        self,
        max_distance: float = 0.0,
    ) -> None:

        if not np.isfinite(max_distance):
            raise QualityError("Max distance must be finite.")

        if max_distance < 0:
            raise QualityError("Max distance cannot be negative.")

        self._max_distance = float(max_distance)

    @property
    def max_distance(self) -> float:
        """Maximum nearest-neighbor distance."""
        return self._max_distance

    def compute(
        self,
        reference: NDArray[np.float64],
        compared: NDArray[np.float64],
    ) -> CloudToCloudResult:
        """
        Compute cloud-to-cloud distances.
        """

        reference = np.asarray(
            reference,
            dtype=np.float64,
        )

        compared = np.asarray(
            compared,
            dtype=np.float64,
        )

        self._validate_cloud(
            reference,
            "Reference",
        )

        self._validate_cloud(
            compared,
            "Compared",
        )

        if _HAS_SCIPY:
            distances = self._compute_kdtree(
                reference,
                compared,
            )

        else:
            distances = self._compute_bruteforce(
                reference,
                compared,
            )

        unmatched = ~np.isfinite(distances)

        if np.any(unmatched):
            unmatched_count = int(np.count_nonzero(unmatched))
            raise QualityError(
                "Cloud-to-Cloud correspondence incomplete: "
                f"{unmatched_count} of {distances.size} reference points "
                "have no nearest neighbor within max_distance."
            )

        # At this point every reference point has a finite correspondence.
        # Do not silently discard unmatched points: doing so would bias
        # quality statistics toward the matched subset.
        std = float(np.std(distances, ddof=1)) if distances.size > 1 else 0.0

        return CloudToCloudResult(
            mean=float(np.mean(distances)),
            std=std,
            minimum=float(np.min(distances)),
            maximum=float(np.max(distances)),
            median=float(np.median(distances)),
            distances=distances,
        )

    @staticmethod
    def _validate_cloud(
        cloud: NDArray[np.float64],
        name: str,
    ) -> None:
        """
        Validate point cloud dimensions.
        """

        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise QualityError(f"{name} cloud must have shape (n, 3).")

        if cloud.shape[0] == 0:
            raise QualityError(f"{name} cloud must not be empty.")

        if not np.isfinite(cloud).all():
            raise QualityError(f"{name} cloud contains NaN or infinite coordinates.")

    def _compute_kdtree(
        self,
        reference: NDArray[np.float64],
        compared: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute nearest-neighbor distances using KD-tree.
        """

        tree = cKDTree(compared)

        max_distance = self._max_distance if self._max_distance > 0 else float("inf")

        result, _ = tree.query(
            reference,
            k=1,
            distance_upper_bound=max_distance,
        )

        return np.asarray(
            result,
            dtype=np.float64,
        )

    def _compute_bruteforce(
        self,
        reference: NDArray[np.float64],
        compared: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        NumPy fallback nearest-neighbor computation.
        """

        distances = np.full(
            reference.shape[0],
            np.inf,
            dtype=np.float64,
        )

        for index, point in enumerate(reference):
            diff = compared - point

            squared = np.sum(
                diff**2,
                axis=1,
            )

            distance = float(np.sqrt(np.min(squared)))

            if self._max_distance == 0 or distance <= self._max_distance:
                distances[index] = distance

        return distances

    def __call__(
        self,
        reference: NDArray[np.float64],
        compared: NDArray[np.float64],
    ) -> CloudToCloudResult:

        return self.compute(
            reference,
            compared,
        )


__all__ = [
    "CloudToCloudDistance",
]
