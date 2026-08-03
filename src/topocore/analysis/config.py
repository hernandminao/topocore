"""
topocore.analysis.config
========================

Global configuration for the TopoCore analysis subsystem.

This module defines configuration dataclasses for the different analysis
modules (distance, volume, profile, visibility, statistics, quality),
allowing fine-grained control over algorithm parameters without modifying
the core logic.

All configuration objects are immutable and follow a consistent pattern.
Defaults are chosen to provide sensible behavior for typical
topographic / engineering workflows.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ============================================================================
# Distance Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class DistanceConfig:
    """
    Configuration for distance calculations.

    Attributes
    ----------
    default_method
        Default distance method.
    ellipsoid
        Ellipsoid name for geodesic calculations.
    default_precision
        Number of decimal places in distance output.
    """

    default_method: str = "euclidean"
    ellipsoid: str = "WGS84"
    default_precision: int = 6

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.default_precision < 0:
            raise ValueError("Distance precision must be greater than or equal to zero.")

        if not self.ellipsoid:
            raise ValueError("Ellipsoid name cannot be empty.")


# ============================================================================
# Volume Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class VolumeConfig:
    """
    Configuration for volume calculations.

    Attributes
    ----------
    default_method
        Default volume calculation method.
    default_precision
        Number of decimal places in volume output.
    """

    default_method: str = "cut_fill"
    default_precision: int = 4

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.default_precision < 0:
            raise ValueError("Volume precision must be greater than or equal to zero.")


# ============================================================================
# Profile Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """
    Configuration for profile generation.

    Attributes
    ----------
    default_method
        Default profile generation method.
    default_interval
        Station interval along profile axis.
    default_width
        Default profile width.
    """

    default_method: str = "longitudinal"
    default_interval: float = 1.0
    default_width: float = 20.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.default_interval <= 0:
            raise ValueError("Profile interval must be greater than zero.")

        if self.default_width <= 0:
            raise ValueError("Profile width must be greater than zero.")


# ============================================================================
# Visibility Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class VisibilityConfig:
    """
    Configuration for visibility analysis.

    Attributes
    ----------
    default_method
        Default visibility method.
    observer_height
        Observer height above terrain.
    target_height
        Target height above terrain.
    earth_curvature_correction
        Whether Earth curvature correction is applied.
    """

    default_method: str = "los"
    observer_height: float = 1.7
    target_height: float = 0.0
    earth_curvature_correction: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.observer_height < 0:
            raise ValueError("Observer height cannot be negative.")

        if self.target_height < 0:
            raise ValueError("Target height cannot be negative.")


# ============================================================================
# Statistics Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class StatisticsConfig:
    """
    Configuration for statistical analysis.

    Attributes
    ----------
    default_method
        Default statistics method.
    percentile_precision
        Number of decimal places for percentile values.
    histogram_bins
        Number of histogram bins.
    """

    default_method: str = "elevation"
    percentile_precision: int = 4
    histogram_bins: int = 10

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.percentile_precision < 0:
            raise ValueError("Percentile precision must be greater than or equal to zero.")

        if self.histogram_bins <= 0:
            raise ValueError("Histogram bins must be greater than zero.")


# ============================================================================
# Quality Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """
    Configuration for quality assessment.

    Attributes
    ----------
    default_method
        Default quality assessment method.
    confidence_level
        Confidence level for statistical intervals.
    max_correspondence_distance
        Maximum correspondence distance for matching.
    """

    default_method: str = "rmse"
    confidence_level: float = 0.95
    max_correspondence_distance: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")

        if self.max_correspondence_distance <= 0:
            raise ValueError("Maximum correspondence distance must be greater than zero.")


# ============================================================================
# Global Analysis Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """
    Master configuration for the analysis subsystem.

    Aggregates all analysis configurations into a single immutable object.
    """

    distance: DistanceConfig = DistanceConfig()
    volume: VolumeConfig = VolumeConfig()
    profile: ProfileConfig = ProfileConfig()
    visibility: VisibilityConfig = VisibilityConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    quality: QualityConfig = QualityConfig()


#: Default analysis configuration instance.
DEFAULT_ANALYSIS_CONFIG: Final[AnalysisConfig] = AnalysisConfig()


__all__ = [
    "DistanceConfig",
    "VolumeConfig",
    "ProfileConfig",
    "VisibilityConfig",
    "StatisticsConfig",
    "QualityConfig",
    "AnalysisConfig",
    "DEFAULT_ANALYSIS_CONFIG",
]
