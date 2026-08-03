"""
topocore.analysis.volume.grid_volume
=====================================

Grid-based volume computation.

Computes volume differences between two regular elevation grids.
Each cell contributes:

    V = cell_area * elevation_difference

Positive values represent cut.
Negative values represent fill.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from topocore.analysis._shared.volume import (
    compute_cut_fill,
    validate_volume_arrays,
)
from topocore.analysis.exceptions import VolumeError
from topocore.analysis.protocols import GriddedSurface
from topocore.analysis.types import VolumeResult

FloatArray = NDArray[np.floating[Any]]


class GridVolume:
    """
    Computes volume between two regular grids.

    Parameters
    ----------
    resolution
        Grid cell size in meters.
    """

    __slots__ = (
        "_resolution",
        "_cell_area",
    )

    def __init__(
        self,
        resolution: float,
    ) -> None:

        if not np.isfinite(resolution):
            raise VolumeError("Resolution must be finite.")

        if resolution <= 0:
            raise VolumeError("Resolution must be positive.")

        self._resolution = float(resolution)

        self._cell_area = self._resolution**2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> float:
        """
        Grid resolution in meters.
        """
        return self._resolution

    @property
    def cell_area(self) -> float:
        """
        Cell area in square meters.
        """
        return self._cell_area

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(
        self,
        existing: FloatArray,
        proposed: FloatArray,
    ) -> VolumeResult:
        """
        Compute grid volume.

        Parameters
        ----------
        existing
            Existing terrain elevations.
        proposed
            Proposed surface elevations.

        Returns
        -------
        VolumeResult
            Cut, fill and net volume.
        """

        validate_volume_arrays(
            existing,
            proposed,
        )

        cut, fill, net = compute_cut_fill(
            existing,
            proposed,
            self._cell_area,
        )

        return VolumeResult(
            cut_volume=cut,
            fill_volume=fill,
            net_volume=net,
            method="grid_volume",
        )

    # ------------------------------------------------------------------
    # DTM integration
    # ------------------------------------------------------------------

    def compute_from_dtm(
        self,
        existing_dtm: GriddedSurface,
        proposed_dtm: GriddedSurface,
    ) -> VolumeResult:
        """
        Compute volume from DTM surfaces.

        Both DTMs must have identical grid geometry.
        """

        if existing_dtm.grid != proposed_dtm.grid:
            raise VolumeError("DTMs must share the same grid geometry.")

        return self.compute(
            existing_dtm.elevations,
            proposed_dtm.elevations,
        )

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        existing: FloatArray,
        proposed: FloatArray,
    ) -> VolumeResult:

        return self.compute(
            existing,
            proposed,
        )


__all__ = [
    "GridVolume",
]
