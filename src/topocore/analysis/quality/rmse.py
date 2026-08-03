"""
topocore.analysis.quality.rmse
===============================

Root Mean Square Error (RMSE) analysis.

Computes RMSE between paired observations.

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
from topocore.analysis.types import RMSEResult

FloatArray = NDArray[np.float64]


class RMSEAnalysis:
    """
    Computes RMSE between paired observations.
    """

    __slots__ = ()

    @staticmethod
    def compute(
        reference: FloatArray,
        observed: FloatArray,
    ) -> RMSEResult:

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

        if reference.size == 0:
            raise QualityError("Input arrays must not be empty.")

        residuals = reference - observed

        if not np.all(np.isfinite(residuals)):
            raise QualityError("Residuals contain NaN or infinite values.")

        if residuals.ndim == 1:
            rmse_z = float(np.sqrt(np.mean(residuals**2)))

            return RMSEResult(
                vertical=rmse_z,
                horizontal=0.0,
                total=rmse_z,
                count=residuals.size,
            )

        if residuals.ndim == 2 and residuals.shape[1] in (2, 3):
            rmse_x = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))

            rmse_y = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))

            horizontal = float(np.sqrt(rmse_x**2 + rmse_y**2))

            if residuals.shape[1] == 3:
                rmse_z = float(np.sqrt(np.mean(residuals[:, 2] ** 2)))

                total = float(np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2))

            else:
                rmse_z = 0.0
                total = horizontal

            return RMSEResult(
                vertical=rmse_z,
                horizontal=horizontal,
                total=total,
                count=residuals.shape[0],
            )

        raise QualityError(f"Unsupported shape {residuals.shape}. Expected 1D, (n,2) or (n,3).")

    @staticmethod
    def compute_3d(
        reference: FloatArray,
        observed: FloatArray,
    ) -> RMSEResult:

        return RMSEAnalysis.compute(
            reference,
            observed,
        )

    @staticmethod
    def componentwise(
        reference_x: FloatArray,
        reference_y: FloatArray,
        reference_z: FloatArray,
        observed_x: FloatArray,
        observed_y: FloatArray,
        observed_z: FloatArray,
    ) -> RMSEResult:

        arrays = [
            np.asarray(reference_x, dtype=np.float64),
            np.asarray(reference_y, dtype=np.float64),
            np.asarray(reference_z, dtype=np.float64),
            np.asarray(observed_x, dtype=np.float64),
            np.asarray(observed_y, dtype=np.float64),
            np.asarray(observed_z, dtype=np.float64),
        ]

        if any(arr.size == 0 for arr in arrays):
            raise QualityError("Component arrays cannot be empty.")

        if any(not np.all(np.isfinite(arr)) for arr in arrays):
            raise QualityError("Component arrays contain invalid values.")

        rx, ry, rz, ox, oy, oz = arrays

        for ref, obs in (
            (rx, ox),
            (ry, oy),
            (rz, oz),
        ):
            if ref.shape != obs.shape:
                raise QualityError("Reference and observed components must have the same shape.")

        rmse_x = float(np.sqrt(np.mean((rx - ox) ** 2)))
        rmse_y = float(np.sqrt(np.mean((ry - oy) ** 2)))
        rmse_z = float(np.sqrt(np.mean((rz - oz) ** 2)))

        return RMSEResult(
            horizontal=float(np.sqrt(rmse_x**2 + rmse_y**2)),
            vertical=rmse_z,
            total=float(np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)),
            count=rx.size,
        )

    def __call__(
        self,
        reference: FloatArray,
        observed: FloatArray,
    ) -> RMSEResult:

        return self.compute(
            reference,
            observed,
        )


__all__ = [
    "RMSEAnalysis",
]
