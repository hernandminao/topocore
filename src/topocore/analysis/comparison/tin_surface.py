"""
topocore.analysis.comparison.tin_surface
=========================================

TIN-to-TIN surface comparison.

Evaluates two triangulated surfaces over a shared grid built from
their overlapping XY domain, then delegates the cut/fill/unchanged
classification to `SurfaceComparison` -- the same primitive used for
gridded-surface comparison -- rather than reimplementing that logic.

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

from topocore.analysis.exceptions import VolumeError
from topocore.analysis.protocols import TriangulatedSurface

from .result import SurfaceComparisonResult
from .surface import SurfaceComparison

FloatArray = NDArray[np.float64]


class TINComparison:
    """
    Compare two triangulated (TIN) surfaces.

    Both surfaces are evaluated over a shared regular grid covering
    their overlapping XY domain (the intersection of the two TINs'
    own `bounds`), then compared with the same `SurfaceComparison`
    used for gridded surfaces -- see that class for the difference
    convention (``proposed - existing``) and tolerance semantics.

    Known limitations
    ------------------
    - `topocore.terrain.tin.TIN` carries no CRS information at all
      (confirmed: no `crs` property, no CRS parameter on
      `from_points()`/`from_mesh()`). This class therefore cannot
      validate that the two TINs share a common CRS -- that check is
      simply not representable at the TIN level in this codebase
      today. Callers are responsible for ensuring both TINs are
      expressed in the same coordinate system before comparing them.
    - `TIN.find_triangle()` (used internally by both `interpolate()`
      and `contains()`) is a documented O(triangle_count) brute-force
      scan (see that method's own docstring). Evaluating a shared
      grid therefore costs O(grid_points x triangle_count) -- fine
      for typical survey-sized TINs, but a real cost to be aware of
      for very large or very fine-resolution comparisons.
    - This class compares two TINs by sampling both at regular grid
      CELL CENTERS over their shared domain, not by computing an
      exact geometric intersection of the two triangulations
      themselves. A cell whose center happens to be interpolable in
      both TINs is treated as fully valid, even though the actual
      overlap of the two hulls within that cell may only be partial
      (or, at fine enough triangle scale relative to the sampling
      resolution, a cell's center could fall in a different local
      triangle than most of the cell's true area). This is an
      accepted approximation for PR20 -- exact triangle-triangle
      intersection, spatial indexing to avoid the brute-force scan
      above, and other sampling-strategy refinements belong to PR21
      (Optimization), not mixed into this PR's scope of "correct
      behavior with a stable contract first."

    Parameters
    ----------
    resolution
        Grid cell size, in meters, used to sample both TINs over
        their shared domain. Must be finite and positive.
    tolerance
        Elevation difference tolerance passed through to the
        underlying `SurfaceComparison`.
    max_grid_cells
        Safety limit on the number of grid cells the shared-domain
        sampling grid may contain, matching the same
        ``max_grid_cells`` guard already used by
        ``topocore.processing.ground.pmf.PMFGroundClassifier`` for an
        analogous risk. Found and fixed in PR20.5: an unvalidated,
        very small ``resolution`` relative to a large shared domain
        previously created an astronomically large grid with no
        error -- confirmed directly: resolution=0.001 over a 20x20m
        domain builds a 20,000 x 20,000 grid (4*10**8 cells), and
        given this class's own documented O(grid_points x
        triangle_count) sampling cost, that combination hung
        indefinitely rather than raising. Must be a positive integer.
    """

    __slots__ = ("_comparison", "_max_grid_cells", "_resolution")

    def __init__(
        self,
        *,
        resolution: float,
        tolerance: float = 0.0,
        max_grid_cells: int = 8_000_000,
    ) -> None:
        if not math.isfinite(resolution):
            raise VolumeError("Resolution must be finite.")

        if resolution <= 0.0:
            raise VolumeError("Resolution must be positive.")

        if isinstance(max_grid_cells, bool) or not isinstance(max_grid_cells, int) or max_grid_cells < 1:
            raise VolumeError(f"max_grid_cells must be an integer >= 1, got {max_grid_cells}.")

        self._resolution = float(resolution)
        self._max_grid_cells = int(max_grid_cells)
        self._comparison = SurfaceComparison(tolerance=tolerance)

    @property
    def resolution(self) -> float:
        """Grid resolution used to sample both TINs."""
        return self._resolution

    @property
    def tolerance(self) -> float:
        """Elevation comparison tolerance."""
        return self._comparison.tolerance

    def compute(
        self,
        existing: TriangulatedSurface,
        proposed: TriangulatedSurface,
    ) -> SurfaceComparisonResult:
        """
        Compare two TIN surfaces over their shared XY domain.

        Parameters
        ----------
        existing
            Existing triangulated surface.
        proposed
            Proposed triangulated surface.

        Returns
        -------
        SurfaceComparisonResult
            Per-cell classification and summary statistics, computed
            over the shared grid.

        Raises
        ------
        VolumeError
            If either TIN has no triangles, the two TINs' bounds do
            not overlap, or no sampled grid cell falls inside both
            hulls.
        """
        if existing.triangle_count <= 0:
            raise VolumeError("Existing TIN contains no triangles.")

        if proposed.triangle_count <= 0:
            raise VolumeError("Proposed TIN contains no triangles.")

        min_x = max(existing.bounds[0], proposed.bounds[0])
        min_y = max(existing.bounds[1], proposed.bounds[1])
        max_x = min(existing.bounds[2], proposed.bounds[2])
        max_y = min(existing.bounds[3], proposed.bounds[3])

        if max_x <= min_x or max_y <= min_y:
            raise VolumeError("Existing and proposed TINs do not share an overlapping XY domain.")

        columns = max(1, math.ceil((max_x - min_x) / self._resolution))
        rows = max(1, math.ceil((max_y - min_y) / self._resolution))

        if rows * columns > self._max_grid_cells:
            raise VolumeError(
                f"Shared-domain grid would contain {rows * columns} cells, "
                f"exceeding max_grid_cells={self._max_grid_cells}. Increase "
                "resolution or max_grid_cells."
            )

        xs = (min_x + (np.arange(columns) + 0.5) * self._resolution).astype(np.float64)
        ys = (min_y + (np.arange(rows) + 0.5) * self._resolution).astype(np.float64)

        existing_grid = self._sample(existing, xs, ys, rows, columns)
        proposed_grid = self._sample(proposed, xs, ys, rows, columns)

        return self._comparison.compute(existing_grid, proposed_grid)

    @staticmethod
    def _sample(
        surface: TriangulatedSurface,
        xs: FloatArray,
        ys: FloatArray,
        rows: int,
        columns: int,
    ) -> FloatArray:
        """
        Evaluate a TIN's elevation over a regular grid.

        `interpolate()` raises `ValueError` for points outside the
        TIN's convex hull (a different NoData convention than the
        NaN-based one gridded surfaces use, e.g.
        `topocore.terrain.dtm.DTM.from_tin()`) -- caught here and
        mapped to NaN, so `SurfaceComparison`'s own NaN-based
        NoData handling applies uniformly to both gridded and TIN
        comparisons.
        """
        values = np.full((rows, columns), np.nan, dtype=np.float64)

        for row in range(rows):
            for col in range(columns):
                try:
                    values[row, col] = surface.interpolate(float(xs[col]), float(ys[row]))
                except ValueError:
                    continue

        return values

    def __call__(
        self,
        existing: TriangulatedSurface,
        proposed: TriangulatedSurface,
    ) -> SurfaceComparisonResult:
        """Compare two TIN surfaces."""
        return self.compute(existing, proposed)


__all__ = [
    "TINComparison",
]
