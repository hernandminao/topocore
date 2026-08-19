"""
topocore.analysis.quality.completeness
======================================

Completeness analysis.

Evaluates the completeness of a point cloud or terrain model
by comparing the coverage against a reference area.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import QualityError
from topocore.analysis.types import CompletenessResult


class CompletenessAnalysis:
    """
    Evaluates spatial completeness of data coverage.

    Parameters
    ----------
    resolution
        Grid cell size (meters).
    """

    __slots__ = ("_resolution",)

    def __init__(
        self,
        resolution: float = 1.0,
    ) -> None:
        if resolution <= 0.0:
            raise QualityError("Resolution must be positive.")

        self._resolution = float(resolution)

    @property
    def resolution(self) -> float:
        return self._resolution

    def compute(
        self,
        points: NDArray[np.float64],
        reference_bbox: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> CompletenessResult:
        """
        Evaluate spatial completeness.

        Parameters
        ----------
        points
            Point coordinates.

        reference_bbox
            (min_x, min_y, max_x, max_y)

        Returns
        -------
        CompletenessResult
        """
        points = np.asarray(
            points,
            dtype=np.float64,
        )

        if points.ndim != 2:
            raise QualityError("Points must be a 2D array.")

        if points.shape[1] not in (2, 3):
            raise QualityError("Points must have shape (n,2) or (n,3).")

        if points.shape[0] == 0:
            raise QualityError("Point array must not be empty.")

        if not np.isfinite(points).all():
            raise QualityError("Point coordinates contain NaN or infinite values.")

        if len(reference_bbox) != 4:
            raise QualityError("Bounding box must contain four values.")

        min_x, min_y, max_x, max_y = reference_bbox

        if not np.isfinite(np.asarray(reference_bbox, dtype=np.float64)).all():
            raise QualityError("Bounding box contains NaN or infinite values.")

        if max_x <= min_x or max_y <= min_y:
            raise QualityError("Bounding box has invalid extent.")

        n_cols = math.ceil((max_x - min_x) / self._resolution)

        n_rows = math.ceil((max_y - min_y) / self._resolution)

        if n_cols <= 0 or n_rows <= 0:
            raise QualityError("Computed grid is empty.")

        coverage = np.zeros(
            (n_rows, n_cols),
            dtype=np.bool_,
        )

        cols = ((points[:, 0] - min_x) / self._resolution).astype(np.int64)

        rows = ((points[:, 1] - min_y) / self._resolution).astype(np.int64)

        mask = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)

        coverage[
            rows[mask],
            cols[mask],
        ] = True

        covered_cells = int(np.count_nonzero(coverage))

        total_cells = n_rows * n_cols

        cell_area = self._resolution**2

        total_area = float(total_cells) * cell_area

        uncovered_area = float(total_cells - covered_cells) * cell_area

        return CompletenessResult(
            coverage_ratio=covered_cells / total_cells,
            uncovered_area=uncovered_area,
            total_area=total_area,
        )

    def compute_from_grid(
        self,
        data_grid: NDArray[np.bool_],
        reference_grid: NDArray[np.bool_],
    ) -> CompletenessResult:
        """
        Evaluate completeness from boolean grids.

        Parameters
        ----------
        data_grid
            Grid indicating where data exists.

        reference_grid
            Grid defining the reference coverage.

        Returns
        -------
        CompletenessResult
        """
        data_grid = np.asarray(
            data_grid,
            dtype=np.bool_,
        )

        reference_grid = np.asarray(
            reference_grid,
            dtype=np.bool_,
        )

        if data_grid.ndim != 2:
            raise QualityError("Data grid must be a 2D array.")

        if reference_grid.ndim != 2:
            raise QualityError("Reference grid must be a 2D array.")

        if data_grid.shape != reference_grid.shape:
            raise QualityError("Grid shapes must match.")

        total = int(np.count_nonzero(reference_grid))

        if total == 0:
            raise QualityError("Reference grid contains no valid cells.")

        covered = int(np.count_nonzero(data_grid & reference_grid))

        cell_area = self._resolution * self._resolution

        return CompletenessResult(
            coverage_ratio=covered / total,
            uncovered_area=float(total - covered) * cell_area,
            total_area=float(total) * cell_area,
        )

    def __call__(
        self,
        points: NDArray[np.float64],
        reference_bbox: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> CompletenessResult:
        """Alias for :meth:`compute`."""
        return self.compute(
            points,
            reference_bbox,
        )


__all__ = [
    "CompletenessAnalysis",
]
