"""
topocore.analysis.quality.precision
====================================

Precision analysis.

Evaluates measurement repeatability from repeated observations.
Computes standard deviation and confidence intervals for measured
coordinates.

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
from scipy.stats import t

from topocore.analysis.exceptions import QualityError
from topocore.analysis.types import PrecisionResult


class PrecisionAnalysis:
    """
    Evaluates measurement precision from repeated observations.

    Parameters
    ----------
    tolerance_x
        Horizontal precision tolerance in X (meters).
    tolerance_y
        Horizontal precision tolerance in Y (meters).
    tolerance_z
        Vertical precision tolerance in Z (meters).
    """

    __slots__ = (
        "_confidence_level",
        "_tol_x",
        "_tol_y",
        "_tol_z",
    )

    def __init__(
        self,
        tolerance_x: float = 0.02,
        tolerance_y: float = 0.02,
        tolerance_z: float = 0.04,
        confidence_level: float = 0.95,
    ) -> None:

        if tolerance_x <= 0:
            raise QualityError("Tolerance X must be positive.")

        if tolerance_y <= 0:
            raise QualityError("Tolerance Y must be positive.")

        if tolerance_z <= 0:
            raise QualityError("Tolerance Z must be positive.")

        if not 0.0 < confidence_level < 1.0:
            raise QualityError("Confidence level must be between 0 and 1.")

        self._tol_x = float(tolerance_x)
        self._tol_y = float(tolerance_y)
        self._tol_z = float(tolerance_z)
        self._confidence_level = float(confidence_level)

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

    @property
    def confidence_level(self) -> float:
        """Confidence level used for statistical intervals."""
        return self._confidence_level

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(
        self,
        observations: NDArray[np.float64],
    ) -> PrecisionResult:
        """
        Evaluate precision from repeated observations.

        Parameters
        ----------
        observations
            Observation matrix:

            - ``(n,2)`` for XY measurements.
            - ``(n,3)`` for XYZ measurements.

        Returns
        -------
        PrecisionResult
            Precision statistics.

        Raises
        ------
        QualityError
            If observations are invalid.
        """

        data = np.asarray(
            observations,
            dtype=np.float64,
        )

        if data.ndim != 2:
            raise QualityError("Observations must be a 2D array.")

        if data.shape[0] < 3:
            raise QualityError("At least 3 observations are required.")

        dimensions = data.shape[1]

        if dimensions not in (2, 3):
            raise QualityError("Observations must have shape (n,2) or (n,3).")

        if not np.isfinite(data).all():
            raise QualityError("Observations contain NaN or infinite values.")

        # Standard deviation sample estimator
        std_x = float(
            np.std(
                data[:, 0],
                ddof=1,
            )
        )

        std_y = float(
            np.std(
                data[:, 1],
                ddof=1,
            )
        )

        if dimensions == 3:
            std_z = float(
                np.std(
                    data[:, 2],
                    ddof=1,
                )
            )
        else:
            std_z = 0.0

        # Confidence interval for the estimated mean deviation.
        #
        # The margin of error is:
        #
        #   t_(1-alpha/2, n-1) * s / sqrt(n)
        #
        # Using ``s`` directly would produce a confidence interval
        # for an individual observation scale, not for the estimated
        # mean represented by this result.
        count = data.shape[0]
        degrees_of_freedom = count - 1
        alpha = 1.0 - self._confidence_level

        t_value = float(
            t.ppf(
                1.0 - alpha / 2.0,
                degrees_of_freedom,
            )
        )

        if dimensions == 3:
            deviation = std_z
        else:
            deviation = math.sqrt(std_x**2 + std_y**2)

        confidence_margin = t_value * deviation / math.sqrt(count)

        confidence_interval = (
            -confidence_margin,
            confidence_margin,
        )

        return PrecisionResult(
            std_dev=deviation,
            confidence_interval=confidence_interval,
            confidence_level=self._confidence_level,
            count=count,
        )

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        observations: NDArray[np.float64],
    ) -> PrecisionResult:
        """
        Compute precision.
        """

        return self.compute(observations)


__all__ = [
    "PrecisionAnalysis",
]
