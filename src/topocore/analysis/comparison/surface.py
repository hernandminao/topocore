"""
topocore.analysis.comparison.surface
=====================================

Surface-to-surface elevation comparison.

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

from topocore.analysis._shared.volume import validate_volume_arrays
from topocore.analysis.exceptions import VolumeError

from .result import SurfaceComparisonResult

FloatArray = NDArray[np.float64]


class SurfaceComparison:
    """
    Compare two gridded terrain surfaces.

    The elevation difference is defined as::

        difference = proposed - existing

    Therefore:

    * difference < -tolerance -> cut (proposed is below existing)
    * difference > +tolerance -> fill (proposed is above existing)
    * abs(difference) <= tolerance -> unchanged

    Validation is delegated to
    `topocore.analysis._shared.volume.validate_volume_arrays` -- the
    same shape/dimensionality/NoData rules already used by
    `CutFillVolume`/`GridVolume`, so a surface accepted by one is
    accepted by the other. NaN cells (in either array) are excluded
    from the comparison rather than propagating or raising -- see
    that function's own docstring for why this matters for real,
    irregularly-bounded DTM surfaces.

    Parameters
    ----------
    tolerance
        Elevation difference tolerance, in meters. Differences with
        absolute value at or below this are classified as
        unchanged rather than cut/fill. Must be non-negative.
    """

    __slots__ = ("_tolerance",)

    def __init__(
        self,
        *,
        tolerance: float = 0.0,
    ) -> None:
        if not math.isfinite(tolerance):
            raise VolumeError("Comparison tolerance must be finite.")

        if tolerance < 0.0:
            raise VolumeError("Comparison tolerance cannot be negative.")

        self._tolerance = float(tolerance)

    @property
    def tolerance(self) -> float:
        """Elevation comparison tolerance."""
        return self._tolerance

    def compute(
        self,
        existing: FloatArray,
        proposed: FloatArray,
    ) -> SurfaceComparisonResult:
        """
        Compare two elevation grids.

        Parameters
        ----------
        existing
            Existing terrain elevations.
        proposed
            Proposed design elevations.

        Returns
        -------
        SurfaceComparisonResult
            Per-cell classification and summary statistics.

        Raises
        ------
        VolumeError
            If the surfaces are invalid or share no valid cells.
        """
        existing_array = np.asarray(existing, dtype=np.float64)
        proposed_array = np.asarray(proposed, dtype=np.float64)

        validate_volume_arrays(existing_array, proposed_array)

        valid = np.isfinite(existing_array) & np.isfinite(proposed_array)

        difference = np.full(existing_array.shape, np.nan, dtype=np.float64)
        difference[valid] = proposed_array[valid] - existing_array[valid]

        cut_mask = valid & (difference < -self._tolerance)
        fill_mask = valid & (difference > self._tolerance)
        unchanged_mask = valid & ~cut_mask & ~fill_mask

        valid_values = difference[valid]

        if valid_values.size == 0:
            raise VolumeError(
                "Surface comparison contains no valid overlapping cells "
                "(existing and proposed have no shared non-NaN cells)."
            )

        return SurfaceComparisonResult(
            difference=difference,
            cut_mask=cut_mask,
            fill_mask=fill_mask,
            unchanged_mask=unchanged_mask,
            valid_cells=int(valid_values.size),
            excluded_cells=int(difference.size - valid_values.size),
            minimum_difference=float(np.min(valid_values)),
            maximum_difference=float(np.max(valid_values)),
            mean_difference=float(np.mean(valid_values)),
        )

    def __call__(
        self,
        existing: FloatArray,
        proposed: FloatArray,
    ) -> SurfaceComparisonResult:
        """Compare two surfaces."""
        return self.compute(existing, proposed)


__all__ = [
    "SurfaceComparison",
]
