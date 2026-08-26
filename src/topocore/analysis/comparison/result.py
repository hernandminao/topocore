"""
topocore.analysis.comparison.result
====================================

Results produced by surface comparison analysis.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SurfaceComparisonResult:
    """
    Result of comparing two terrain surfaces.

    Parameters
    ----------
    difference
        Elevation difference ``proposed - existing`` (NaN at
        excluded/NoData cells).
    cut_mask
        Cells where the proposed surface is below the existing
        surface (beyond tolerance).
    fill_mask
        Cells where the proposed surface is above the existing
        surface (beyond tolerance).
    unchanged_mask
        Valid cells whose difference falls within tolerance.
    valid_cells
        Number of valid comparison cells.
    excluded_cells
        Number of excluded NoData cells.
    minimum_difference
        Minimum valid elevation difference.
    maximum_difference
        Maximum valid elevation difference.
    mean_difference
        Mean valid elevation difference.
    """

    difference: FloatArray
    cut_mask: BoolArray
    fill_mask: BoolArray
    unchanged_mask: BoolArray

    valid_cells: int
    excluded_cells: int

    minimum_difference: float
    maximum_difference: float
    mean_difference: float

    @property
    def cut_cells(self) -> int:
        """Number of cut cells."""
        return int(np.count_nonzero(self.cut_mask))

    @property
    def fill_cells(self) -> int:
        """Number of fill cells."""
        return int(np.count_nonzero(self.fill_mask))

    @property
    def unchanged_cells(self) -> int:
        """Number of unchanged cells."""
        return int(np.count_nonzero(self.unchanged_mask))

    @property
    def total_cells(self) -> int:
        """
        Total number of cells in the compared grid, valid or not.

        Added in PR20.6, corrected before close: this must be
        derived from ``difference.size`` (the actual grid), not from
        ``valid_cells + excluded_cells``. Deriving it from those two
        fields would make the documented invariant
        ``valid_cells + excluded_cells == total_cells`` a tautology
        -- always true by construction, even if a
        `SurfaceComparisonResult` were ever built inconsistently
        (e.g. by a future bug in `SurfaceComparison`/`TINComparison`,
        or a hand-constructed instance in a test). Deriving
        ``total_cells`` independently from the grid itself means the
        invariant is a genuine, checkable property of the result,
        not something this property enforces by definition.
        """
        return int(self.difference.size)


__all__ = [
    "SurfaceComparisonResult",
]
