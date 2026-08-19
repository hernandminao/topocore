"""
topocore.analysis.quality.hausdorff
===================================

Hausdorff distance analysis.

Computes the Hausdorff distance between two point sets. The
Hausdorff distance is the maximum of all the distances from a
point in one set to the closest point in the other set:

    H(A, B) = max(h(A, B), h(B, A))

where h(A, B) = max_{a in A} min_{b in B} d(a, b).

Also provides the directed Hausdorff distance and the modified
Hausdorff distance using a percentile instead of the maximum.

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
from topocore.analysis.types import HausdorffResult

try:
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True

except ImportError:
    _HAS_SCIPY = False


class HausdorffDistance:
    """
    Computes Hausdorff distances between two point sets.

    Parameters
    ----------
    percentile
        Percentile to use for the modified Hausdorff distance.
        100 computes the classical Hausdorff distance.
    """

    __slots__ = ("_percentile",)

    def __init__(
        self,
        percentile: float = 100.0,
    ) -> None:

        if not (0.0 < percentile <= 100.0):
            raise QualityError("Percentile must be in (0, 100].")

        self._percentile = float(percentile)

    @property
    def percentile(self) -> float:
        """Percentile used for the directed Hausdorff distance."""
        return self._percentile

    def compute(
        self,
        set_a: NDArray[np.float64],
        set_b: NDArray[np.float64],
    ) -> HausdorffResult:
        """
        Compute the Hausdorff distance between two point sets.

        Parameters
        ----------
        set_a
            First point set with shape ``(n, 3)``.
        set_b
            Second point set with shape ``(m, 3)``.

        Returns
        -------
        HausdorffResult
            Directed and symmetric Hausdorff distances.

        Raises
        ------
        QualityError
            If either point set is invalid.
        """

        set_a = np.asarray(
            set_a,
            dtype=np.float64,
        )

        set_b = np.asarray(
            set_b,
            dtype=np.float64,
        )

        if set_a.ndim != 2 or set_a.shape[1] != 3:
            raise QualityError("Set A must have shape (n, 3).")

        if set_b.ndim != 2 or set_b.shape[1] != 3:
            raise QualityError("Set B must have shape (m, 3).")

        if set_a.shape[0] == 0:
            raise QualityError("Set A must not be empty.")

        if set_b.shape[0] == 0:
            raise QualityError("Set B must not be empty.")

        if not np.isfinite(set_a).all():
            raise QualityError("Set A contains NaN or infinite coordinates.")

        if not np.isfinite(set_b).all():
            raise QualityError("Set B contains NaN or infinite coordinates.")

        h_ab = self._directed_hausdorff(
            set_a,
            set_b,
        )

        h_ba = self._directed_hausdorff(
            set_b,
            set_a,
        )

        return HausdorffResult(
            directed_forward=h_ab,
            directed_backward=h_ba,
            hausdorff=max(h_ab, h_ba),
        )

    def _directed_hausdorff(
        self,
        source: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> float:
        """
        Compute the directed Hausdorff distance.

        h(A, B) = max_a min_b d(a, b)

        or a percentile-based variant when
        ``percentile < 100``.
        """

        if source.shape[1] != target.shape[1]:
            raise QualityError("Point sets must have the same dimensionality.")

        if _HAS_SCIPY:
            tree = cKDTree(target)

            result, _ = tree.query(
                source,
                k=1,
            )

            distances = np.asarray(
                result,
                dtype=np.float64,
            )

        else:
            distances = np.empty(
                source.shape[0],
                dtype=np.float64,
            )

            for index, point in enumerate(source):
                diff = target - point

                distance = float(
                    np.sqrt(
                        np.min(
                            np.sum(
                                diff**2,
                                axis=1,
                            )
                        )
                    )
                )

                distances[index] = distance

        return float(
            np.percentile(
                distances,
                self._percentile,
            )
        )

    def __call__(
        self,
        set_a: NDArray[np.float64],
        set_b: NDArray[np.float64],
    ) -> HausdorffResult:
        """
        Compute the Hausdorff distance.

        This method allows the class instance to be used as a callable.
        """

        return self.compute(
            set_a,
            set_b,
        )


__all__ = [
    "HausdorffDistance",
]
