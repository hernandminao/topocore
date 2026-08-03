"""
topocore.analysis.volume.cut_fill
==================================

Cut-and-fill volume computation between two elevation surfaces.

Computes removed and added material volumes by comparing an
existing terrain surface against a proposed design surface.

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


class CutFillVolume:
    """
    Computes cut and fill volumes between two surfaces.

    Parameters
    ----------
    cell_area
        Area of each raster cell in square meters.
    """

    __slots__ = ("_cell_area",)

    def __init__(
        self,
        cell_area: float,
    ) -> None:

        if not np.isfinite(cell_area):
            raise VolumeError("Cell area must be finite.")

        if cell_area <= 0:
            raise VolumeError("Cell area must be positive.")

        self._cell_area = float(cell_area)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
        Compute cut/fill volume between two elevation grids.

        Parameters
        ----------
        existing
            Existing terrain elevations.
        proposed
            Proposed design elevations.

        Returns
        -------
        VolumeResult
            Cut, fill and net volumes.

        Raises
        ------
        VolumeError
            If surfaces are incompatible.
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
            method="cut_fill",
        )

    # ------------------------------------------------------------------
    # DTM Integration
    # ------------------------------------------------------------------

    def compute_with_dtm(
        self,
        existing_dtm: GriddedSurface,
        proposed_dtm: GriddedSurface,
    ) -> VolumeResult:
        """
        Compute volume using two DTM surfaces.

        Both DTMs must share identical grid geometry.
        """

        if existing_dtm.grid != proposed_dtm.grid:
            raise VolumeError("DTMs must share the same grid geometry.")

        resolution = float(existing_dtm.resolution)

        return CutFillVolume(resolution**2).compute(
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
        """
        Compute cut/fill volume.
        """

        return self.compute(
            existing,
            proposed,
        )


__all__ = [
    "CutFillVolume",
]
