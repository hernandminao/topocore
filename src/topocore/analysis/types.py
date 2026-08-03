"""
topocore.analysis.types
=======================

Type aliases and dataclass result types used throughout
the analysis module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Array aliases
# ============================================================================

#: A one-dimensional NumPy array of 64-bit floats.
FloatArray1D: TypeAlias = NDArray[np.float64]

#: A two-dimensional NumPy array of 64-bit floats.
FloatArray2D: TypeAlias = NDArray[np.float64]

#: A one-dimensional NumPy array of boolean values.
BoolArray1D: TypeAlias = NDArray[np.bool_]

#: A one-dimensional NumPy array of integer indices.
IntArray1D: TypeAlias = NDArray[np.int64]


# ============================================================================
# Distance
# ============================================================================


@dataclass(frozen=True, slots=True)
class DistanceResult:
    """
    Result of distance computation.

    Parameters
    ----------
    value:
        Distance value in meters.
    method:
        Computation method.
    dimension:
        Distance dimension.
    """

    value: float
    method: str
    dimension: str

    @property
    def meters(self) -> float:
        """Return distance in meters."""
        return self.value


# ============================================================================
# Volume
# ============================================================================


class VolumeMethod(StrEnum):
    """
    Supported volume computation methods.
    """

    CUT_FILL = "cut_fill"
    PRISMOIDAL = "prismoidal"
    AVERAGE_END_AREA = "average_end_area"
    TIN_VOLUME = "tin_volume"
    GRID_VOLUME = "grid_volume"


@dataclass(frozen=True, slots=True)
class VolumeResult:
    """
    Volume computation result.
    """

    cut_volume: float
    fill_volume: float
    net_volume: float
    method: str

    @property
    def cubic_meters(self) -> float:
        """Return total displaced volume."""
        return self.cut_volume + self.fill_volume


# ============================================================================
# Profile result types
# ============================================================================


class ProfileType(StrEnum):
    """
    Profile generation types.
    """

    LONGITUDINAL = "longitudinal"
    TRANSVERSAL = "transversal"
    CROSS_SECTION = "cross_section"
    MULTI = "multi"


@dataclass(frozen=True, slots=True)
class ProfilePoint:
    """
    Single point in a terrain profile.
    """

    station: float
    x: float
    y: float
    z: float
    offset: float = 0.0


@dataclass(slots=True)
class ProfileResult:
    """
    Profile generation result.
    """

    points: list[ProfilePoint] = field(default_factory=list)

    profile_type: ProfileType = ProfileType.LONGITUDINAL

    axis_length: float = 0.0

    def distances(self) -> list[float]:
        """Return station distances."""
        return [point.station for point in self.points]

    def elevations(self) -> list[float]:
        """Return profile elevations."""
        return [point.z for point in self.points]

    def offsets(self) -> list[float]:
        """Return profile offsets."""
        return [point.offset for point in self.points]


# ============================================================================
# Visibility result types
# ============================================================================


class VisibilityType(StrEnum):
    """
    Visibility analysis methods.
    """

    LINE_OF_SIGHT = "los"
    VIEWSHED = "viewshed"
    INTERVISIBILITY = "intervisibility"


@dataclass(frozen=True, slots=True)
class LOSResult:
    """
    Line-of-Sight computation result.

    Attributes
    ----------
    visible:
        Whether target is visible.
    distance:
        LOS distance in meters.
    obstacles:
        Blocking triangle indexes.
    elevation_at_target:
        Terrain elevation at target.
    clearance:
        Minimum vertical clearance.
    """

    visible: bool

    distance: float

    obstacles: tuple[int, ...]

    elevation_at_target: float = 0.0

    clearance: float = 0.0


@dataclass(slots=True)
class ViewshedResult:
    """
    Viewshed computation result.
    """

    visibility_map: NDArray[np.bool_] = field(
        default_factory=lambda: np.empty(
            (0, 0),
            dtype=np.bool_,
        )
    )

    visible_count: int = 0

    total_count: int = 0

    @property
    def visibility_ratio(self) -> float:
        """
        Ratio of visible cells.
        """
        if self.total_count <= 0:
            return 0.0

        return self.visible_count / self.total_count

    def visible_points(
        self,
    ) -> NDArray[np.bool_]:
        """
        Return flattened visibility mask.
        """
        return self.visibility_map.ravel()


@dataclass(frozen=True, slots=True)
class IntervisibilityResult:
    """
    Pairwise visibility result.
    """

    visibility_matrix: NDArray[np.bool_] = field(
        default_factory=lambda: np.empty(
            (0, 0),
            dtype=np.bool_,
        )
    )

    visible_pairs: int = 0

    total_pairs: int = 0

    @property
    def visibility_ratio(self) -> float:
        """
        Ratio of visible pairs.
        """
        if self.total_pairs <= 0:
            return 0.0

        return self.visible_pairs / self.total_pairs


# ============================================================================
# Statistics result types
# ============================================================================


@dataclass(frozen=True, slots=True)
class ElevationStats:
    """
    Elevation statistics.

    Attributes
    ----------
    minimum
        Minimum elevation.
    maximum
        Maximum elevation.
    mean
        Mean elevation.
    std
        Standard deviation of elevation.
    median
        Median elevation.
    range
        Elevation range (max - min).
    count
        Number of points / cells.
    """

    minimum: float
    maximum: float
    mean: float
    std: float
    median: float
    range: float
    count: int

    def percentiles(
        self,
        values: FloatArray1D,
        percentiles: list[float],
    ) -> dict[float, float]:
        """
        Compute percentiles from original elevation values.

        Parameters
        ----------
        values:
            Original elevation array.
        percentiles:
            Percentile values from 0 to 100.

        Returns
        -------
        dict:
            Mapping percentile -> value.
        """

        valid = values[np.isfinite(values)]

        if valid.size == 0:
            return {}

        return {
            p: float(
                np.percentile(
                    valid,
                    p,
                )
            )
            for p in percentiles
        }


@dataclass(frozen=True, slots=True)
class SlopeStats:
    """
    Slope statistics.

    Attributes
    ----------
    minimum
        Minimum slope (degrees).
    maximum
        Maximum slope (degrees).
    mean
        Mean slope (degrees).
    std
        Standard deviation of slope.
    histogram_bins
        List of bin edges.
    histogram_counts
        Count per bin.
    """

    minimum: float
    maximum: float
    mean: float
    std: float
    histogram_bins: list[float] = field(default_factory=list)
    histogram_counts: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AreaStats:
    """
    Terrain area statistics.
    """

    total_area: float

    projected_area: float

    surface_area: float

    count: int


@dataclass(frozen=True, slots=True)
class DensityStats:
    """
    Point density statistics.

    Attributes
    ----------
    mean_density
        Mean points per square meter.
    minimum_density
        Minimum local density.
    maximum_density
        Maximum local density.
    std_density
        Standard deviation of density.
    density_map
        2D density grid.
    """

    mean_density: float
    minimum_density: float
    maximum_density: float
    std_density: float
    density_map: FloatArray2D = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))


@dataclass(frozen=True, slots=True)
class DistributionStats:
    """
    Statistical distribution description.
    """

    bins: list[float] = field(default_factory=list)

    counts: list[int] = field(default_factory=list)

    cumulative: list[float] = field(default_factory=list)

    skewness: float = 0.0

    kurtosis: float = 0.0


StatisticsResult: TypeAlias = ElevationStats | SlopeStats | AreaStats | DensityStats | DistributionStats

# ============================================================================
# Quality result types
# ============================================================================


@dataclass(frozen=True, slots=True)
class RMSEResult:
    """
    Root Mean Square Error metrics.
    """

    vertical: float

    horizontal: float

    total: float

    count: int

    @property
    def rmse_3d(self) -> float:
        """
        Return 3D RMSE.
        """
        return self.total


@dataclass(frozen=True, slots=True)
class GPSControlResult:
    """
    GPS control point validation result.
    """

    control_point_count: int

    points_passing: int

    points_failing: int

    max_deviation: float

    mean_deviation: float

    tolerance: float


@dataclass(frozen=True, slots=True)
class PrecisionResult:
    """
    Precision estimation result.
    """

    std_dev: float

    confidence_interval: tuple[float, float]

    confidence_level: float

    count: int


@dataclass(frozen=True, slots=True)
class RegistrationQualityResult:
    """
    Registration quality metrics.
    """

    fitness: float

    rmse: float

    mean_distance: float

    max_distance: float


@dataclass(frozen=True, slots=True)
class CompletenessResult:
    """
    Coverage completeness metric.
    """

    coverage_ratio: float

    uncovered_area: float

    total_area: float


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """
    Classification correctness metrics.
    """

    true_positives: int

    false_positives: int

    false_negatives: int

    precision: float

    recall: float

    f1_score: float


# ============================================================================
# Point cloud distance metrics
# ============================================================================


@dataclass(frozen=True, slots=True)
class CloudToCloudResult:
    """
    Cloud-to-cloud distance statistics.
    """

    mean: float

    std: float

    minimum: float

    maximum: float

    median: float

    distances: FloatArray1D = field(
        default_factory=lambda: np.empty(
            0,
            dtype=np.float64,
        )
    )


@dataclass(frozen=True, slots=True)
class CloudToMeshResult:
    """
    Cloud-to-mesh distance statistics.
    """

    mean: float

    std: float

    minimum: float

    maximum: float

    median: float

    distances: FloatArray1D = field(
        default_factory=lambda: np.empty(
            0,
            dtype=np.float64,
        )
    )


@dataclass(frozen=True, slots=True)
class HausdorffResult:
    """
    Hausdorff distance result.
    """

    directed_forward: float

    directed_backward: float

    hausdorff: float

    @property
    def symmetric(self) -> float:
        """
        Return symmetric Hausdorff distance.
        """
        return self.hausdorff


@dataclass(frozen=True, slots=True)
class ChamferResult:
    """
    Chamfer distance result.
    """

    directed_forward: float

    directed_backward: float

    chamfer: float

    @property
    def symmetric(self) -> float:
        """
        Return symmetric Chamfer distance.
        """
        return self.chamfer


@dataclass(frozen=True, slots=True)
class QualitySummary:
    """
    Generic quality evaluation summary.
    """

    metric: str
    value: float
    units: str
    threshold: float | None = None
    passed: bool | None = None


__all__ = [
    # ---------------------------------------------------------------------
    # Array aliases
    # ---------------------------------------------------------------------
    "FloatArray1D",
    "FloatArray2D",
    "BoolArray1D",
    "IntArray1D",
    # ---------------------------------------------------------------------
    # Enumerations
    # ---------------------------------------------------------------------
    "VolumeMethod",
    "ProfileType",
    "VisibilityType",
    # ---------------------------------------------------------------------
    # Distance
    # ---------------------------------------------------------------------
    "DistanceResult",
    # ---------------------------------------------------------------------
    # Volume
    # ---------------------------------------------------------------------
    "VolumeResult",
    # ---------------------------------------------------------------------
    # Profile
    # ---------------------------------------------------------------------
    "ProfilePoint",
    "ProfileResult",
    # ---------------------------------------------------------------------
    # Visibility
    # ---------------------------------------------------------------------
    "LOSResult",
    "ViewshedResult",
    "IntervisibilityResult",
    # ---------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------
    "ElevationStats",
    "SlopeStats",
    "AreaStats",
    "DensityStats",
    "DistributionStats",
    "StatisticsResult",
    # ---------------------------------------------------------------------
    # Quality
    # ---------------------------------------------------------------------
    "RMSEResult",
    "GPSControlResult",
    "PrecisionResult",
    "RegistrationQualityResult",
    "CompletenessResult",
    "CorrectnessResult",
    # ---------------------------------------------------------------------
    # Point cloud distances
    # ---------------------------------------------------------------------
    "CloudToCloudResult",
    "CloudToMeshResult",
    "HausdorffResult",
    "ChamferResult",
]
