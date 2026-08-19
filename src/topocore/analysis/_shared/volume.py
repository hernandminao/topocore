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

    Notes
    -----
    NaN cells ARE allowed here (a common, legitimate case: a DTM
    grid extending slightly beyond its source data's convex hull,
    e.g. ``topocore.terrain.dtm.DTM.from_tin()``, correctly leaves
    those cells as NaN -- see that module's own PR19 fix). Rejecting
    any grid containing NaN made cut/fill volume computation
    unusable on essentially any real, irregularly-bounded terrain
    surface. ``compute_cut_fill`` excludes NaN cells from the sums
    rather than failing outright -- this function only rejects
    genuinely unusable input (empty, mismatched shape, wrong
    dimensionality, or infinite values, which are never legitimate
    NoData markers in this codebase and likely indicate a real
    upstream error).
    """

    if existing.size == 0:
        raise VolumeError("Existing surface contains no elevation values.")

    if proposed.size == 0:
        raise VolumeError("Proposed surface contains no elevation values.")

    if existing.shape != proposed.shape:
        raise VolumeError(f"Shape mismatch: {existing.shape} != {proposed.shape}")

    if existing.ndim != 2:
        raise VolumeError("Elevation arrays must be 2-dimensional grids.")

    if np.isinf(existing).any():
        raise VolumeError("Existing surface contains infinite elevations.")

    if np.isinf(proposed).any():
        raise VolumeError("Proposed surface contains infinite elevations.")

    if np.isnan(existing).all():
        raise VolumeError("Existing surface has no valid (non-NaN) elevations.")

    if np.isnan(proposed).all():
        raise VolumeError("Proposed surface has no valid (non-NaN) elevations.")


def compute_cut_fill(
    existing: FloatArray,
    proposed: FloatArray,
    cell_area: float,
) -> tuple[float, float, float, int, int]:
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

    Returns
    -------
    tuple
        cut, fill, net volumes, valid_cells, excluded_cells.

    Notes
    -----
    Cells that are NaN in EITHER array (e.g. outside a DTM's convex
    hull) are excluded from the volume sums rather than propagating
    NaN or raising -- see ``validate_volume_arrays``'s docstring.
    ``excluded_cells`` tells the caller how much of the grid had no
    overlapping data, so a near-total exclusion (e.g. mismatched or
    barely-overlapping surfaces) is visible rather than silently
    returning a near-zero volume from a handful of valid cells.
    """

    if not np.isfinite(cell_area):
        raise VolumeError("Cell area must be finite.")

    if cell_area <= 0:
        raise VolumeError("Cell area must be positive.")

    valid_mask = np.isfinite(existing) & np.isfinite(proposed)
    valid_cells = int(np.count_nonzero(valid_mask))
    excluded_cells = int(existing.size - valid_cells)

    diff = np.where(valid_mask, existing - proposed, 0.0)

    cut = np.maximum(diff, 0.0)
    fill = np.maximum(-diff, 0.0)

    cut_volume = float(np.sum(cut) * cell_area)

    fill_volume = float(np.sum(fill) * cell_area)

    return (
        cut_volume,
        fill_volume,
        cut_volume - fill_volume,
        valid_cells,
        excluded_cells,
    )


__all__ = [
    "FloatArray",
    "compute_cut_fill",
    "validate_volume_arrays",
]
