"""
topocore.analysis.quality.gps_control
======================================

GPS control point quality assessment.

Evaluates the accuracy of a survey by comparing surveyed control
points against known reference coordinates. Produces a GPS control
result with per-component RMSE, mean errors, and a pass/fail
assessment against configurable tolerances.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import QualityError
from topocore.analysis.types import GPSControlResult


class GPSControlAnalysis:
    """
    Evaluates GPS control point accuracy.

    Parameters
    ----------
    tolerance_x
        Horizontal tolerance in X (meters).
    tolerance_y
        Horizontal tolerance in Y (meters).
    tolerance_z
        Vertical tolerance in Z (meters).
    """

    __slots__ = (
        "_tol_x",
        "_tol_y",
        "_tol_z",
    )

    def __init__(
        self,
        tolerance_x: float = 0.05,
        tolerance_y: float = 0.05,
        tolerance_z: float = 0.10,
    ) -> None:

        for name, value in {
            "tolerance_x": tolerance_x,
            "tolerance_y": tolerance_y,
            "tolerance_z": tolerance_z,
        }.items():
            if not np.isfinite(value):
                raise QualityError(f"{name} must be finite.")

            if value <= 0:
                raise QualityError(f"{name} must be positive.")

        self._tol_x = float(tolerance_x)
        self._tol_y = float(tolerance_y)
        self._tol_z = float(tolerance_z)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tolerance_x(self) -> float:
        return self._tol_x

    @property
    def tolerance_y(self) -> float:
        return self._tol_y

    @property
    def tolerance_z(self) -> float:
        return self._tol_z

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(
        self,
        reference: NDArray[np.float64],
        observed: NDArray[np.float64],
    ) -> GPSControlResult:
        """
        Evaluate GPS control accuracy.

        Parameters
        ----------
        reference
            Reference XYZ coordinates.
        observed
            Observed XYZ coordinates.

        Returns
        -------
        GPSControlResult
            GPS accuracy assessment.
        """

        reference = np.asarray(
            reference,
            dtype=np.float64,
        )

        observed = np.asarray(
            observed,
            dtype=np.float64,
        )

        if reference.shape != observed.shape:
            raise QualityError(f"Shape mismatch: reference {reference.shape} vs observed {observed.shape}.")

        if reference.ndim != 2 or reference.shape[1] != 3:
            raise QualityError("Expected coordinate arrays with shape (n,3).")

        if reference.shape[0] == 0:
            raise QualityError("Input arrays must not be empty.")

        if not np.isfinite(reference).all() or not np.isfinite(observed).all():
            raise QualityError("Coordinates contain NaN or infinite values.")

        residuals = reference - observed

        distances = np.sqrt(
            np.sum(
                residuals**2,
                axis=1,
            )
        )

        points_passing = int(
            np.sum(
                (np.abs(residuals[:, 0]) <= self._tol_x)
                & (np.abs(residuals[:, 1]) <= self._tol_y)
                & (np.abs(residuals[:, 2]) <= self._tol_z)
            )
        )

        count = reference.shape[0]

        return GPSControlResult(
            control_point_count=count,
            points_passing=points_passing,
            points_failing=count - points_passing,
            max_deviation=float(np.max(distances)),
            mean_deviation=float(np.mean(distances)),
            tolerance=self._tol_z,
        )

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        reference: NDArray[np.float64],
        observed: NDArray[np.float64],
    ) -> GPSControlResult:

        return self.compute(
            reference,
            observed,
        )


__all__ = [
    "GPSControlAnalysis",
]
