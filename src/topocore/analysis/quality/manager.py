"""
topocore.analysis.quality.manager
==================================

Unified quality analysis facade.

Provides a single entry point for quality assessment computations
using any supported method.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.config import (
    DEFAULT_ANALYSIS_CONFIG,
    QualityConfig,
)
from topocore.analysis.exceptions import QualityError
from topocore.analysis.protocols import TriangulatedSurface
from topocore.analysis.types import (
    ChamferResult,
    CloudToCloudResult,
    CloudToMeshResult,
    CompletenessResult,
    CorrectnessResult,
    GPSControlResult,
    HausdorffResult,
    PrecisionResult,
    RegistrationQualityResult,
    RMSEResult,
)

from .c2c import CloudToCloudDistance
from .c2m import CloudToMeshDistance
from .chamfer import ChamferDistance
from .completeness import CompletenessAnalysis
from .correctness import CorrectnessAnalysis
from .gps_control import GPSControlAnalysis
from .hausdorff import HausdorffDistance
from .precision import PrecisionAnalysis
from .registration import RegistrationQuality
from .rmse import RMSEAnalysis

QualityResult: TypeAlias = (
    RMSEResult
    | GPSControlResult
    | PrecisionResult
    | RegistrationQualityResult
    | CompletenessResult
    | CorrectnessResult
    | CloudToCloudResult
    | CloudToMeshResult
    | HausdorffResult
    | ChamferResult
)

DispatchMethod: TypeAlias = Callable[..., QualityResult]


class QualityMethod:
    """Supported quality analysis methods."""

    RMSE = "rmse"
    GPS_CONTROL = "gps_control"
    PRECISION = "precision"
    REGISTRATION = "registration"
    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    CLOUD_TO_CLOUD = "c2c"
    CLOUD_TO_MESH = "c2m"
    HAUSDORFF = "hausdorff"
    CHAMFER = "chamfer"


_VALID_METHODS = frozenset(
    {
        QualityMethod.RMSE,
        QualityMethod.GPS_CONTROL,
        QualityMethod.PRECISION,
        QualityMethod.REGISTRATION,
        QualityMethod.COMPLETENESS,
        QualityMethod.CORRECTNESS,
        QualityMethod.CLOUD_TO_CLOUD,
        QualityMethod.CLOUD_TO_MESH,
        QualityMethod.HAUSDORFF,
        QualityMethod.CHAMFER,
    }
)


class QualityAnalysis:
    """
    Unified quality analysis manager.

    Parameters
    ----------
    config
        Quality-specific configuration.
    """

    __slots__ = (
        "_config",
        "_dispatch",
        "_method",
        "_tolerance",
    )

    def __init__(
        self,
        *,
        config: QualityConfig | None = None,
        method: str | None = None,
    ) -> None:
        self._config = config or DEFAULT_ANALYSIS_CONFIG.quality
        self._method = method or self._config.default_method

        if self._method not in _VALID_METHODS:
            raise QualityError(f"Unknown quality method: '{self._method}'.")

        self._tolerance = self._config.max_correspondence_distance

        self._dispatch: dict[str, DispatchMethod] = {
            QualityMethod.RMSE: self.rmse,
            QualityMethod.GPS_CONTROL: self.gps_control,
            QualityMethod.PRECISION: self.precision,
            QualityMethod.REGISTRATION: self.registration,
            QualityMethod.COMPLETENESS: self.completeness,
            QualityMethod.CORRECTNESS: self.correctness,
            QualityMethod.CLOUD_TO_CLOUD: self.cloud_to_cloud,
            QualityMethod.CLOUD_TO_MESH: self.cloud_to_mesh,
            QualityMethod.HAUSDORFF: self.hausdorff,
            QualityMethod.CHAMFER: self.chamfer,
        }

    @property
    def method(self) -> str:
        """Configured quality analysis method."""
        return self._method

    @property
    def config(self) -> QualityConfig:
        """Quality analysis configuration."""
        return self._config

    def rmse(
        self,
        reference: NDArray[np.float64],
        observed: NDArray[np.float64],
    ) -> RMSEResult:
        """Compute RMSE between paired observations."""
        return RMSEAnalysis.compute(
            reference,
            observed,
        )

    def gps_control(
        self,
        reference: NDArray[np.float64],
        observed: NDArray[np.float64],
        tolerance_x: float | None = None,
        tolerance_y: float | None = None,
        tolerance_z: float | None = None,
    ) -> GPSControlResult:
        """Evaluate GPS control point accuracy."""
        tx = 0.05 if tolerance_x is None else tolerance_x
        ty = 0.05 if tolerance_y is None else tolerance_y
        tz = 0.10 if tolerance_z is None else tolerance_z

        return GPSControlAnalysis(
            tx,
            ty,
            tz,
        ).compute(
            reference,
            observed,
        )

    def precision(
        self,
        observations: NDArray[np.float64],
        tolerance_x: float | None = None,
        tolerance_y: float | None = None,
        tolerance_z: float | None = None,
    ) -> PrecisionResult:
        """Evaluate measurement precision."""
        tx = 0.02 if tolerance_x is None else tolerance_x
        ty = 0.02 if tolerance_y is None else tolerance_y
        tz = 0.04 if tolerance_z is None else tolerance_z

        return PrecisionAnalysis(
            tx,
            ty,
            tz,
            confidence_level=self._config.confidence_level,
        ).compute(observations)

    def registration(
        self,
        residuals: NDArray[np.float64],
        tolerance: float | None = None,
    ) -> RegistrationQualityResult:
        """Evaluate registration quality."""
        tol = self._tolerance if tolerance is None else tolerance

        return RegistrationQuality(tol).compute(residuals)

    def completeness(
        self,
        points: NDArray[np.float64],
        reference_bbox: tuple[float, float, float, float],
        resolution: float = 1.0,
    ) -> CompletenessResult:
        """Evaluate spatial completeness."""
        return CompletenessAnalysis(resolution).compute(
            points,
            reference_bbox,
        )

    def correctness(
        self,
        model_grid: NDArray[np.float64],
        reference_grid: NDArray[np.float64],
        tolerance: float | None = None,
    ) -> CorrectnessResult:
        """Evaluate model correctness."""
        tol = self._tolerance if tolerance is None else tolerance

        return CorrectnessAnalysis(tol).compute(
            model_grid,
            reference_grid,
        )

    def cloud_to_cloud(
        self,
        reference: NDArray[np.float64],
        compared: NDArray[np.float64],
        max_distance: float = 0.0,
    ) -> CloudToCloudResult:
        """Compute Cloud-to-Cloud distances."""
        return CloudToCloudDistance(max_distance).compute(
            reference,
            compared,
        )

    def cloud_to_mesh(
        self,
        pointcloud: NDArray[np.float64],
        tin: TriangulatedSurface,
        max_distance: float = 0.0,
    ) -> CloudToMeshResult:
        """
        Compute Cloud-to-Mesh distances.

        Parameters
        ----------
        pointcloud
            Point cloud coordinates.
        tin
            Triangulated surface reference.
        max_distance
            Maximum correspondence distance.

        Returns
        -------
        CloudToMeshResult
            Cloud-to-Mesh quality metrics.
        """
        return CloudToMeshDistance(max_distance).compute(
            pointcloud,
            tin,
        )

    def hausdorff(
        self,
        set_a: NDArray[np.float64],
        set_b: NDArray[np.float64],
        percentile: float = 100.0,
    ) -> HausdorffResult:
        """Compute Hausdorff distance."""
        return HausdorffDistance(percentile).compute(
            set_a,
            set_b,
        )

    def chamfer(
        self,
        set_a: NDArray[np.float64],
        set_b: NDArray[np.float64],
    ) -> ChamferResult:
        """Compute Chamfer distance."""
        return ChamferDistance().compute(
            set_a,
            set_b,
        )

    def compute(
        self,
        *args: Any,
        method: str | None = None,
        **kwargs: Any,
    ) -> QualityResult:
        """
        Compute quality metrics using selected method.

        Parameters
        ----------
        *args
            Arguments forwarded to the selected quality method.

        method
            Optional method override.

        **kwargs
            Keyword arguments forwarded to the selected quality method.

        Returns
        -------
        QualityResult
            Quality analysis result.

        Raises
        ------
        QualityError
            If the requested method is not supported.
        """
        target = method or self._method

        if target not in _VALID_METHODS:
            raise QualityError(f"Unknown quality method: '{target}'.")

        analyzer = self._dispatch[target]

        return analyzer(
            *args,
            **kwargs,
        )

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> QualityResult:
        """Execute configured quality analysis."""
        return self.compute(
            *args,
            **kwargs,
        )


__all__ = [
    "QualityAnalysis",
    "QualityMethod",
]
