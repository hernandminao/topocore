"""
topocore.analysis
=================

Analysis module for TopoCore.

Provides computation tools for distances, volumes, profiles, visibility,
statistics, and quality assessment of terrain models and point clouds.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.analysis.config import (
    DEFAULT_ANALYSIS_CONFIG,
    AnalysisConfig,
    DistanceConfig,
    ProfileConfig,
    QualityConfig,
    StatisticsConfig,
    VisibilityConfig,
    VolumeConfig,
)
from topocore.analysis.exceptions import (
    AnalysisError,
    DistanceError,
    ProfileError,
    QualityError,
    StatisticsError,
    VisibilityError,
    VolumeError,
)
from topocore.analysis.protocols import (
    Calculable,
    CoordinateTransformer,
    CRSType,
    Generable,
    GriddedSurface,
    Measurable,
    PointCloudData,
    TerrainSurface,
    TriangulatedSurface,
)
from topocore.analysis.types import (
    # Statistics
    AreaStats,
    # Array aliases
    BoolArray1D,
    # Quality
    ChamferResult,
    CloudToCloudResult,
    CloudToMeshResult,
    CompletenessResult,
    CorrectnessResult,
    DensityStats,
    # Distance
    DistanceResult,
    DistributionStats,
    ElevationStats,
    FloatArray1D,
    FloatArray2D,
    GPSControlResult,
    HausdorffResult,
    IntArray1D,
    # Visibility
    IntervisibilityResult,
    LOSResult,
    PrecisionResult,
    # Profile
    ProfilePoint,
    ProfileResult,
    ProfileType,
    RegistrationQualityResult,
    RMSEResult,
    SlopeStats,
    ViewshedResult,
    VisibilityType,
    # Volume
    VolumeMethod,
    VolumeResult,
)

__version__ = "0.1.0"


__all__ = [
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    "AnalysisConfig",
    "DEFAULT_ANALYSIS_CONFIG",
    "DistanceConfig",
    "ProfileConfig",
    "QualityConfig",
    "StatisticsConfig",
    "VisibilityConfig",
    "VolumeConfig",
    # ------------------------------------------------------------------
    # Exceptions
    # ------------------------------------------------------------------
    "AnalysisError",
    "DistanceError",
    "ProfileError",
    "QualityError",
    "StatisticsError",
    "VisibilityError",
    "VolumeError",
    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------
    "Calculable",
    "CoordinateTransformer",
    "CRSType",
    "Generable",
    "GriddedSurface",
    "Measurable",
    "PointCloudData",
    "TerrainSurface",
    "TriangulatedSurface",
    # ------------------------------------------------------------------
    # Array type aliases
    # ------------------------------------------------------------------
    "BoolArray1D",
    "FloatArray1D",
    "FloatArray2D",
    "IntArray1D",
    # ------------------------------------------------------------------
    # Distance types
    # ------------------------------------------------------------------
    "DistanceResult",
    # ------------------------------------------------------------------
    # Volume types
    # ------------------------------------------------------------------
    "VolumeMethod",
    "VolumeResult",
    # ------------------------------------------------------------------
    # Profile types
    # ------------------------------------------------------------------
    "ProfilePoint",
    "ProfileResult",
    "ProfileType",
    # ------------------------------------------------------------------
    # Visibility types
    # ------------------------------------------------------------------
    "LOSResult",
    "ViewshedResult",
    "IntervisibilityResult",
    "VisibilityType",
    # ------------------------------------------------------------------
    # Statistics types
    # ------------------------------------------------------------------
    "AreaStats",
    "DensityStats",
    "DistributionStats",
    "ElevationStats",
    "SlopeStats",
    # ------------------------------------------------------------------
    # Quality types
    # ------------------------------------------------------------------
    "ChamferResult",
    "CloudToCloudResult",
    "CloudToMeshResult",
    "CompletenessResult",
    "CorrectnessResult",
    "GPSControlResult",
    "HausdorffResult",
    "PrecisionResult",
    "RegistrationQualityResult",
    "RMSEResult",
]
