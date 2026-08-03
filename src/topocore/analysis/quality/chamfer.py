"""
topocore.analysis.quality.chamfer
=================================

Chamfer distance analysis.

Computes the Chamfer distance between two point sets. The Chamfer
distance is the average of the minimum distances from each point in
one set to its nearest neighbor in the other set.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import QualityError
from topocore.analysis.types import ChamferResult

try:
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class ChamferDistance:
    """
    Computes Chamfer distances between two point sets.
    """

    __slots__ = ()

    def compute(
        self,
        set_a: NDArray[np.float64],
        set_b: NDArray[np.float64],
    ) -> ChamferResult:
        """
        Compute the Chamfer distance.

        Parameters
        ----------
        set_a
            First point set.
        set_b
            Second point set.

        Returns
        -------
        ChamferResult
            Chamfer distance metrics.

        Raises
        ------
        QualityError
            If the inputs are invalid.
        """
        set_a = np.asarray(set_a, dtype=np.float64)
        set_b = np.asarray(set_b, dtype=np.float64)

        if set_a.ndim != 2:
            raise QualityError("Set A must be a 2D array.")

        if set_b.ndim != 2:
            raise QualityError("Set B must be a 2D array.")

        if set_a.shape[0] == 0:
            raise QualityError("Set A must not be empty.")

        if set_b.shape[0] == 0:
            raise QualityError("Set B must not be empty.")

        if set_a.shape[1] != set_b.shape[1]:
            raise QualityError("Point sets must have the same dimensionality.")

        if set_a.shape[1] not in (2, 3):
            raise QualityError("Point sets must have shape (n,2) or (n,3).")

        forward = self._average_min_distance(
            set_a,
            set_b,
        )

        backward = self._average_min_distance(
            set_b,
            set_a,
        )

        return ChamferResult(
            directed_forward=forward,
            directed_backward=backward,
            chamfer=(forward + backward) / 2.0,
        )

    @staticmethod
    def _average_min_distance(
        source: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> float:
        """
        Compute the average nearest-neighbor distance from one point
        set to another.
        """
        distances = ChamferDistance._min_distances(
            source,
            target,
        )

        return float(np.mean(distances))

    @staticmethod
    def _min_distances(
        source: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the nearest-neighbor distance from every point in
        ``source`` to ``target``.
        """
        if _HAS_SCIPY:
            tree = cKDTree(target)

            result, _ = tree.query(
                source,
                k=1,
            )

            return cast(
                NDArray[np.float64],
                np.asarray(result, dtype=np.float64),
            )

        distances = np.empty(
            source.shape[0],
            dtype=np.float64,
        )

        for i, point in enumerate(source):
            diff = target - point

            distances[i] = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))

        return distances

    def __call__(
        self,
        set_a: NDArray[np.float64],
        set_b: NDArray[np.float64],
    ) -> ChamferResult:
        return self.compute(
            set_a,
            set_b,
        )


__all__ = [
    "ChamferDistance",
]
