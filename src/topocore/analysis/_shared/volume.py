"""
topocore.analysis._shared.volume
================================

Shared utilities for volume computations.

Centralizes validation and cut/fill calculations used by:
- grid volume
- cut/fill analysis
- terrain volume methods

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

from topocore.analysis.exceptions import VolumeError

FloatArray = NDArray[np.floating[Any]]


def validate_volume_arrays(
    existing: FloatArray,
    proposed: FloatArray,
) -> None:
    """
    Validate elevation arrays.

    Parameters
    ----------
    existing
        Existing terrain elevations.
    proposed
        Proposed terrain elevations.

    Raises
    ------
    VolumeError
        If arrays are invalid.
    """

    if existing.size == 0:
        raise VolumeError("Existing surface contains no elevation values.")

    if proposed.size == 0:
        raise VolumeError("Proposed surface contains no elevation values.")

    if existing.shape != proposed.shape:
        raise VolumeError(f"Shape mismatch: {existing.shape} != {proposed.shape}")

    if existing.ndim != 2:
        raise VolumeError("Elevation arrays must be 2-dimensional grids.")

    if not np.isfinite(existing).all():
        raise VolumeError("Existing surface contains invalid elevations.")

    if not np.isfinite(proposed).all():
        raise VolumeError("Proposed surface contains invalid elevations.")


def compute_cut_fill(
    existing: FloatArray,
    proposed: FloatArray,
    cell_area: float,
) -> tuple[float, float, float]:
    """
    Compute cut/fill volumes.

    Parameters
    ----------
    existing
        Existing terrain.
    proposed
        Proposed terrain.
    cell_area
        Cell area in square meters.
    method
        Volume method identifier.

    Returns
    -------
    tuple
        cut, fill, net volumes.
    """

    if not np.isfinite(cell_area):
        raise VolumeError("Cell area must be finite.")

    if cell_area <= 0:
        raise VolumeError("Cell area must be positive.")

    diff = existing - proposed

    cut = np.maximum(diff, 0.0)
    fill = np.maximum(-diff, 0.0)

    cut_volume = float(np.sum(cut) * cell_area)

    fill_volume = float(np.sum(fill) * cell_area)

    return (
        cut_volume,
        fill_volume,
        cut_volume - fill_volume,
    )


__all__ = [
    "FloatArray",
    "validate_volume_arrays",
    "compute_cut_fill",
]
