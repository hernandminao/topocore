"""
topocore.analysis.visibility.intervisibility
=============================================

Intervisibility analysis.

Computes pairwise visibility between terrain points using
Line-of-Sight analysis over a triangulated terrain surface.

The generated visibility matrix is symmetric:

V(i,j) == V(j,i)

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np

from topocore.analysis.exceptions import VisibilityError
from topocore.analysis.protocols import TriangulatedSurface
from topocore.analysis.types import IntervisibilityResult

from .los import LineOfSight

# ============================================================================
# Constants
# ============================================================================

_DEFAULT_OBSERVER_HEIGHT = 1.7
_DEFAULT_SAMPLE_COUNT = 100


class Intervisibility:
    """
    Computes pairwise intervisibility between points.

    Parameters
    ----------
    observer_height
        Height above terrain for observers.
    num_samples
        LOS samples per pair.
    earth_curvature
        Enable earth curvature correction.
    """

    __slots__ = (
        "_observer_height",
        "_num_samples",
        "_earth_curvature",
    )

    def __init__(
        self,
        observer_height: float = _DEFAULT_OBSERVER_HEIGHT,
        num_samples: int = _DEFAULT_SAMPLE_COUNT,
        earth_curvature: bool = True,
    ) -> None:

        if observer_height < 0:
            raise VisibilityError("Observer height cannot be negative.")

        if num_samples < 2:
            raise VisibilityError("num_samples must be at least 2.")

        self._observer_height = float(observer_height)
        self._num_samples = int(num_samples)
        self._earth_curvature = bool(earth_curvature)

    @property
    def observer_height(self) -> float:
        """Observer height."""
        return self._observer_height

    def compute(
        self,
        points: list[tuple[float, float]],
        tin: TriangulatedSurface,
    ) -> IntervisibilityResult:
        """
        Compute visibility matrix between all points.
        """

        self._validate_points(
            points,
            tin,
        )

        los = LineOfSight(
            observer_height=self._observer_height,
            target_height=self._observer_height,
            earth_curvature=self._earth_curvature,
            num_samples=self._num_samples,
        )

        matrix = self._compute_matrix(
            points,
            tin,
            los,
        )

        visible_pairs = int(np.count_nonzero(np.triu(matrix, k=1)))

        total_pairs = len(points) * (len(points) - 1) // 2

        return IntervisibilityResult(
            visibility_matrix=matrix,
            visible_pairs=visible_pairs,
            total_pairs=total_pairs,
        )

    @staticmethod
    def _validate_points(
        points: list[tuple[float, float]],
        tin: TriangulatedSurface,
    ) -> None:
        """
        Validate input points.
        """

        if len(points) < 2:
            raise VisibilityError("Intervisibility requires at least 2 points.")

        for index, point in enumerate(points):
            if not tin.contains(
                point[0],
                point[1],
            ):
                raise VisibilityError(f"Point {index} is outside the TIN.")

    @staticmethod
    def _compute_matrix(
        points: list[tuple[float, float]],
        tin: TriangulatedSurface,
        los: LineOfSight,
    ) -> np.ndarray:
        """
        Compute symmetric visibility matrix.
        """

        count = len(points)

        matrix = np.zeros(
            (count, count),
            dtype=bool,
        )

        for i in range(count):
            for j in range(
                i + 1,
                count,
            ):
                visible = los.compute(
                    points[i],
                    points[j],
                    tin,
                ).visible

                matrix[i, j] = visible
                matrix[j, i] = visible

        return matrix

    def visibility_network(
        self,
        points: list[tuple[float, float]],
        tin: TriangulatedSurface,
    ) -> list[tuple[int, int]]:
        """
        Return visible point pairs.
        """

        result = self.compute(
            points,
            tin,
        )

        return [
            (i, j)
            for i in range(len(points))
            for j in range(
                i + 1,
                len(points),
            )
            if result.visibility_matrix[i, j]
        ]

    def __call__(
        self,
        points: list[tuple[float, float]],
        tin: TriangulatedSurface,
    ) -> IntervisibilityResult:
        """Execute intervisibility analysis."""
        return self.compute(
            points,
            tin,
        )


__all__ = ["Intervisibility"]
