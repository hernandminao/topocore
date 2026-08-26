"""
topocore.analysis.comparison.cut_fill
======================================

Surface comparison combined with cut/fill volume analysis.

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
from topocore.analysis.types import VolumeResult
from topocore.analysis.volume.cut_fill import CutFillVolume

from .result import SurfaceComparisonResult
from .surface import SurfaceComparison

FloatArray = NDArray[np.float64]


class SurfaceCutFill:
    """
    Combined surface comparison and cut/fill volume analysis.

    Runs `SurfaceComparison` (per-cell ΔZ classification) and
    `CutFillVolume` (cut/fill/net volumes) over the same pair of
    elevation grids, so a caller gets both the spatial picture and
    the volumetric summary from one call, without either analysis
    duplicating the other's math -- `SurfaceComparison` answers "what
    changed", `CutFillVolume` answers "how much material that
    represents" (see `topocore.analysis.volume`).

    Parameters
    ----------
    cell_area
        Area of each raster cell in square meters. Must be finite
        and positive.
    tolerance
        Elevation difference tolerance for the comparison's cut/
        fill/unchanged classification, in meters. Does not affect
        the volume computation (which always integrates every
        non-zero difference, per `CutFillVolume`'s own contract).
    """

    __slots__ = ("_comparison", "_volume")

    def __init__(
        self,
        *,
        cell_area: float,
        tolerance: float = 0.0,
    ) -> None:
        if not math.isfinite(cell_area):
            raise VolumeError("Cell area must be finite.")

        if cell_area <= 0:
            raise VolumeError("Cell area must be positive.")

        self._comparison = SurfaceComparison(tolerance=tolerance)
        self._volume = CutFillVolume(cell_area)

    @property
    def tolerance(self) -> float:
        """Elevation comparison tolerance, in meters."""
        return self._comparison.tolerance

    @property
    def cell_area(self) -> float:
        """Cell area in square meters."""
        return self._volume.cell_area

    def compute(
        self,
        existing: FloatArray,
        proposed: FloatArray,
    ) -> tuple[SurfaceComparisonResult, VolumeResult]:
        """
        Compare surfaces and compute cut/fill volumes.

        Parameters
        ----------
        existing
            Existing terrain elevations.
        proposed
            Proposed design elevations.

        Returns
        -------
        tuple
            ``(comparison, volume)`` -- the per-cell comparison and
            the aggregate cut/fill/net volumes.
        """
        comparison = self._comparison.compute(existing, proposed)
        volume = self._volume.compute(existing, proposed)

        return comparison, volume

    def __call__(
        self,
        existing: FloatArray,
        proposed: FloatArray,
    ) -> tuple[SurfaceComparisonResult, VolumeResult]:
        """Compare surfaces and compute cut/fill volumes."""
        return self.compute(existing, proposed)


__all__ = [
    "SurfaceCutFill",
]
