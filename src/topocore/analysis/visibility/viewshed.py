"""
topocore.analysis.visibility.viewshed
======================================

Viewshed analysis.

Computes the visible area from an observer position across a
triangulated terrain surface. The terrain is discretized into a
regular grid, and for each grid cell the Line-of-Sight is checked
from the observer to the cell center. The result is a boolean
visibility map.

Performance
-----------
The search grid is clamped to the intersection of the observer's
search radius and the TIN's actual bounds, instead of being built
purely around the observer (which could include cells guaranteed to
fall outside the mesh). Grid coordinates and their distance to the
observer are computed as vectorized NumPy arrays, and cells beyond
``max_distance`` are pruned by that array *before* the remaining
loop, which is the only part that must stay per-cell since it calls
into ``TriangulatedSurface.contains`` and ``LineOfSight.compute``
(neither of which is vectorized in the underlying API). That
remaining loop iterates over a flat list of candidate cells, which
makes it straightforward to swap for a parallel map
(``concurrent.futures`` / ``joblib``) later, since each cell's
visibility check is independent.

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

from topocore.analysis.exceptions import VisibilityError
from topocore.analysis.protocols import TriangulatedSurface
from topocore.analysis.types import ViewshedResult

from .los import LineOfSight

_DEFAULT_OBSERVER_HEIGHT = 1.7
_DEFAULT_RESOLUTION = 5.0
_DEFAULT_MAX_DISTANCE = 0.0
_DEFAULT_NUM_SAMPLES = 50


class Viewshed:
    """
    Computes viewsheds from an observer position over a TIN.

    Parameters
    ----------
    observer_height
        Height of the observer above terrain (meters).
    resolution
        Grid cell size for the visibility map (meters).
    max_distance
        Maximum visibility distance (meters).
        Zero means unlimited.
    earth_curvature
        Apply Earth curvature correction.
    num_samples
        Number of LOS samples per cell.
    """

    __slots__ = (
        "_observer_height",
        "_resolution",
        "_max_distance",
        "_earth_curvature",
        "_num_samples",
    )

    def __init__(
        self,
        observer_height: float = _DEFAULT_OBSERVER_HEIGHT,
        resolution: float = _DEFAULT_RESOLUTION,
        max_distance: float = _DEFAULT_MAX_DISTANCE,
        *,
        earth_curvature: bool = True,
        num_samples: int = _DEFAULT_NUM_SAMPLES,
    ) -> None:

        if observer_height < 0:
            raise VisibilityError("Observer height cannot be negative.")

        if resolution <= 0:
            raise VisibilityError("Resolution must be positive.")

        if max_distance < 0:
            raise VisibilityError("Max distance cannot be negative.")

        if num_samples < 2:
            raise VisibilityError("num_samples must be at least 2.")

        self._observer_height = float(observer_height)
        self._resolution = float(resolution)
        self._max_distance = float(max_distance)
        self._earth_curvature = bool(earth_curvature)
        self._num_samples = int(num_samples)

    @property
    def observer_height(self) -> float:
        """Observer height."""
        return self._observer_height

    @property
    def resolution(self) -> float:
        """Grid resolution."""
        return self._resolution

    @property
    def max_distance(self) -> float:
        """Maximum visibility distance."""
        return self._max_distance

    def compute(
        self,
        observer: tuple[float, float],
        tin: TriangulatedSurface,
    ) -> ViewshedResult:
        """
        Compute the visibility map.

        Parameters
        ----------
        observer
            Observer position ``(x, y)``.
        tin
            Triangulated terrain surface.

        Returns
        -------
        ViewshedResult
            Visibility grid and statistics.
        """

        if not tin.contains(
            observer[0],
            observer[1],
        ):
            raise VisibilityError("Observer is outside the TIN.")

        bounds = tin.bounds

        (
            grid_min_x,
            grid_min_y,
            _,
            _,
            n_cols,
            n_rows,
        ) = self._build_grid_extent(
            observer,
            bounds,
        )

        if n_rows <= 0 or n_cols <= 0:
            raise VisibilityError("Viewshed grid has zero dimensions.")

        xx, yy, in_range = self._grid_coordinates_and_range_mask(
            observer,
            grid_min_x,
            grid_min_y,
            n_cols,
            n_rows,
        )

        visibility_map = np.zeros(
            (n_rows, n_cols),
            dtype=bool,
        )

        los = LineOfSight(
            observer_height=self._observer_height,
            target_height=0.0,
            earth_curvature=self._earth_curvature,
            num_samples=self._num_samples,
        )

        candidate_rows, candidate_cols = np.nonzero(in_range)

        visible_count = 0
        checked_count = 0

        for row, col in zip(
            candidate_rows,
            candidate_cols,
            strict=True,
        ):
            x = float(xx[row, col])
            y = float(yy[row, col])

            visible = self._check_cell(
                observer,
                x,
                y,
                tin,
                los,
            )

            if visible is None:
                continue

            checked_count += 1

            visibility_map[row, col] = visible

            if visible:
                visible_count += 1

        return ViewshedResult(
            visibility_map=visibility_map,
            visible_count=visible_count,
            total_count=checked_count,
        )

    def __call__(
        self,
        observer: tuple[float, float],
        tin: TriangulatedSurface,
    ) -> ViewshedResult:
        return self.compute(observer, tin)

    def _build_grid_extent(
        self,
        observer: tuple[float, float],
        bounds: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> tuple[
        float,
        float,
        float,
        float,
        int,
        int,
    ]:

        if self._max_distance > 0:
            half_range = self._max_distance
        else:
            half_range = max(
                abs(bounds[0] - observer[0]),
                abs(bounds[2] - observer[0]),
                abs(bounds[1] - observer[1]),
                abs(bounds[3] - observer[1]),
            )

        min_x = max(
            observer[0] - half_range,
            bounds[0],
        )

        min_y = max(
            observer[1] - half_range,
            bounds[1],
        )

        max_x = min(
            observer[0] + half_range,
            bounds[2],
        )

        max_y = min(
            observer[1] + half_range,
            bounds[3],
        )

        cols = int(math.ceil((max_x - min_x) / self._resolution)) + 1

        rows = int(math.ceil((max_y - min_y) / self._resolution)) + 1

        return (
            min_x,
            min_y,
            max_x,
            max_y,
            cols,
            rows,
        )

    def _grid_coordinates_and_range_mask(
        self,
        observer: tuple[float, float],
        grid_min_x: float,
        grid_min_y: float,
        n_cols: int,
        n_rows: int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]:

        cols = grid_min_x + np.arange(n_cols) * self._resolution

        rows = grid_min_y + np.arange(n_rows) * self._resolution

        xx, yy = np.meshgrid(
            cols,
            rows,
        )

        if self._max_distance <= 0:
            return (
                xx,
                yy,
                np.ones(
                    xx.shape,
                    dtype=bool,
                ),
            )

        distance = np.hypot(
            xx - observer[0],
            yy - observer[1],
        )

        return (
            xx,
            yy,
            distance <= self._max_distance,
        )

    @staticmethod
    def _check_cell(
        observer: tuple[float, float],
        x: float,
        y: float,
        tin: TriangulatedSurface,
        los: LineOfSight,
    ) -> bool | None:

        if not tin.contains(
            x,
            y,
        ):
            return None

        return los.compute(
            observer,
            (x, y),
            tin,
        ).visible


__all__ = ["Viewshed"]
