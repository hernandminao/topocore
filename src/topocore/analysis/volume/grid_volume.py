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
HernÃ¡n Mina

License
-------
MIT
"""

from __future__ import annotations

import math
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
        "_cell_area",
        "_resolution",
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

        cut, fill, net, valid_cells, excluded_cells = compute_cut_fill(
            existing,
            proposed,
            self._cell_area,
        )

        return VolumeResult(
            cut_volume=cut,
            fill_volume=fill,
            net_volume=net,
            method="grid_volume",
            valid_cells=valid_cells,
            excluded_cells=excluded_cells,
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

        Both DTMs must have identical grid geometry, and that
        geometry's resolution must match this instance's own
        `resolution` (see `Raises` below).
        """

        if existing_dtm.grid != proposed_dtm.grid:
            raise VolumeError("DTMs must share the same grid geometry.")

        # Found and fixed in PR20: this previously used self._cell_area
        # (fixed at construction time) unconditionally, never checking
        # it against the DTMs' own actual resolution -- confirmed
        # directly: a GridVolume(resolution=1.0) fed real DTMs at
        # resolution=2.0 silently computed a cut volume exactly 4x
        # too small (cell_area off by resolution-squared), with no
        # error or warning. Unlike CutFillVolume.compute_with_dtm()
        # (which sidesteps this by always deriving a fresh cell_area
        # from the DTM's own resolution), this class's own resolution
        # is a real, meaningful instance property (exposed via a
        # public `resolution`/`cell_area` property, unlike
        # CutFillVolume) -- so silently overriding it here would
        # contradict what the instance claims to be configured for.
        # Fixed by validating the two agree and failing loudly
        # instead, per HernÃ¡n's explicit choice over the alternative
        # (silently deriving from the DTM, matching CutFillVolume).
        if not math.isclose(self._resolution, existing_dtm.resolution):
            raise VolumeError(
                f"GridVolume resolution ({self._resolution}) does not match "
                f"the DTM's resolution ({existing_dtm.resolution})."
            )

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
