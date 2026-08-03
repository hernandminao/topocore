"""
topocore.analysis.quality.registration
======================================

Registration quality assessment.

Evaluates the quality of point cloud or model registration by
computing residual statistics after alignment.

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
from topocore.analysis.types import RegistrationQualityResult


class RegistrationQuality:
    """
    Evaluates registration quality from alignment residuals.

    Parameters
    ----------
    tolerance
        Maximum acceptable residual distance (meters).
    """

    __slots__ = ("_tolerance",)

    def __init__(
        self,
        tolerance: float = 0.05,
    ) -> None:
        if tolerance <= 0.0:
            raise QualityError("Tolerance must be positive.")

        self._tolerance = float(tolerance)

    @property
    def tolerance(self) -> float:
        """Maximum accepted residual distance."""
        return self._tolerance

    def compute(
        self,
        residuals: NDArray[np.float64],
    ) -> RegistrationQualityResult:
        """
        Evaluate registration quality.

        Parameters
        ----------
        residuals
            Residual distances.

            Accepted shapes:

            * ``(n,)``
            * ``(n,3)``

        Returns
        -------
        RegistrationQualityResult

        Raises
        ------
        QualityError
            If the input is invalid.
        """
        residuals = np.asarray(
            residuals,
            dtype=np.float64,
        )

        if residuals.size == 0:
            raise QualityError("Residuals array must not be empty.")

        if residuals.ndim == 1:
            distances = np.abs(residuals)

        elif residuals.ndim == 2 and residuals.shape[1] == 3:
            distances = np.linalg.norm(
                residuals,
                axis=1,
            )

        else:
            raise QualityError(f"Unsupported residual shape {residuals.shape}. Expected (n,) or (n, 3).")

        valid = distances[np.isfinite(distances)]

        if valid.size == 0:
            raise QualityError("No finite residual values.")

        fitness = float(np.count_nonzero(valid <= self._tolerance) / valid.size)

        rmse = float(np.sqrt(np.mean(valid * valid)))

        return RegistrationQualityResult(
            fitness=fitness,
            rmse=rmse,
            mean_distance=float(np.mean(valid)),
            max_distance=float(np.max(valid)),
        )

    def __call__(
        self,
        residuals: NDArray[np.float64],
    ) -> RegistrationQualityResult:
        """Alias for :meth:`compute`."""
        return self.compute(residuals)


__all__ = [
    "RegistrationQuality",
]
