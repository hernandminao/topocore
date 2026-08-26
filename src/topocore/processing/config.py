"""
topocore.processing.config
==========================

Global configuration for the TopoCore processing subsystem.

This module defines configuration dataclasses for the different processing
modules (neighbors, filters, normals, classification, etc.), allowing
fine-grained control over algorithm parameters without modifying the
core logic.

All configuration objects are immutable and follow a consistent pattern.
Defaults are chosen to provide sensible behavior for typical
topographic/LiDAR point clouds (i.e., metric units, moderate densities).

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

from topocore.math.config import DEFAULT_MATH_CONFIG, MathConfig

# ============================================================================
# Neighbor Search Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class NeighborConfig:
    """
    Configuration for spatial neighbor search.

    Attributes
    ----------
    knn_default
        Default number of nearest neighbors to return.
    radius_default
        Default radius for radius-based searches.
    use_adaptive
        Whether to adapt the search radius/k based on local point density.
    adaptive_factor
        Multiplicative factor for adaptive search. For KNN, this scales the
        number of neighbors; for radius, it scales the radius.
    max_neighbors
        Maximum number of neighbors to return, as a safety cap.
    workers
        Number of worker threads scipy.spatial.cKDTree uses internally
        for batched queries (PR21.2). Default (1) preserves EXACTLY
        the pre-PR21 single-threaded behavior -- this is opt-in, never
        forced. -1 uses all available CPU cores. Every query point's
        neighbor search is fully independent of every other's, so
        parallelizing cannot change the numerical result, only
        wall-clock time -- confirmed by
        topocore.processing.neighbors.kdtree's own regression suite.
    """

    knn_default: int = 8
    radius_default: float = 1.0
    use_adaptive: bool = False
    adaptive_factor: float = 1.5
    max_neighbors: int = 128
    workers: int = 1


# ============================================================================
# Outlier Filter Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class OutlierConfig:
    """
    Configuration for outlier removal filters.

    Attributes
    ----------
    statistical_k
        Number of neighbors for statistical outlier detection.
    statistical_std_ratio
        Standard deviation multiplier threshold.
    radius_radius
        Radius for radius-based outlier detection.
    radius_min_neighbors
        Minimum number of neighbors within radius.
    min_points
        Minimum number of points that must remain after filtering.
    """

    statistical_k: int = 8
    statistical_std_ratio: float = 1.0
    radius_radius: float = 1.0
    radius_min_neighbors: int = 4
    min_points: int = 3


# ============================================================================
# Normal Estimation Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class NormalConfig:
    """
    Configuration for surface normal estimation.

    Attributes
    ----------
    knn
        Number of neighbors for PCA-based normal estimation.
    method
        Estimation method: "pca", "weighted_pca", "robust_pca", or "mls".
    orient_to_viewpoint
        If True, flip normals to point towards a given viewpoint.
    """

    knn: int = 10
    method: str = "pca"
    orient_to_viewpoint: bool = False


# ============================================================================
# Classification Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """
    Configuration for point classification.

    Attributes
    ----------
    method
        Classification method: "rule_based", "random_forest", etc.
    ground_cell_size
        Cell size for grid-based ground estimation.
    ground_max_height
        Maximum height above ground for a point to be classified as ground.
    noise_radius
        Radius for noise detection.
    noise_min_neighbors
        Minimum neighbors within noise_radius.
    curvature_k
        Number of neighbors for curvature estimation.
    curvature_threshold
        Curvature threshold for vegetation/building discrimination.
    low_vegetation_max_height
        Height threshold between low and high vegetation.
    """

    method: str = "rule_based"
    ground_cell_size: float = 1.0
    ground_max_height: float = 0.2
    noise_radius: float = 1.0
    noise_min_neighbors: int = 4
    curvature_k: int = 10
    curvature_threshold: float = 0.02
    low_vegetation_max_height: float = 0.5


# ============================================================================
# Sampling Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """
    Configuration for point cloud downsampling.

    Attributes
    ----------
    method
        Sampling method: "random", "uniform", or "voxel".
    voxel_size
        Voxel size for voxel-based downsampling.
    seed
        Random seed for reproducibility (for random sampling).
    """

    method: str = "voxel"
    voxel_size: float = 0.5
    seed: int | None = None


# ============================================================================
# Registration Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class RegistrationConfig:
    """
    Configuration for point cloud registration (ICP and variants).

    Attributes
    ----------
    method
        Registration method: "icp", "point_to_point", "point_to_plane", "gicp".
    max_iterations
        Maximum number of iterations.
    convergence_criteria
        Relative change in fitness / RMSE to stop early.
    max_correspondence_distance
        Maximum distance for a point to be considered a match.
    transformation_epsilon
        Minimum transformation change to stop early.
    """

    method: str = "point_to_point"
    max_iterations: int = 50
    convergence_criteria: float = 1e-6
    max_correspondence_distance: float = 1.0
    transformation_epsilon: float = 1e-8


# ============================================================================
# Segmentation Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """
    Configuration for point cloud segmentation.

    Attributes
    ----------
    method
        Segmentation method: "region_growing", "dbscan", "connected_components".
    min_cluster_size
        Minimum number of points for a valid cluster.
    max_cluster_size
        Maximum number of points for a valid cluster.
    eps
        Epsilon (radius) for DBSCAN.
    min_samples
        Minimum samples for DBSCAN core point.
    """

    method: str = "region_growing"
    min_cluster_size: int = 10
    max_cluster_size: int = 1_000_000
    eps: float = 0.5
    min_samples: int = 5


# ============================================================================
# Global Processing Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """
    Master configuration for the processing subsystem.

    This configuration object aggregates all sub-configurations, providing a
    single entry point for customizing the behavior of TopoCore's processing
    pipeline. Defaults are designed for general-purpose point cloud processing.

    Attributes
    ----------
    neighbor
        Configuration for neighbor search operations.
    outlier
        Configuration for outlier removal filters.
    normal
        Configuration for normal estimation.
    classifier
        Configuration for point classification.
    sampling
        Configuration for point cloud downsampling.
    registration
        Configuration for point cloud registration.
    segmentation
        Configuration for point cloud segmentation.
    math
        Configuration for numerical tolerance and precision.
    """

    neighbor: NeighborConfig = NeighborConfig()
    outlier: OutlierConfig = OutlierConfig()
    normal: NormalConfig = NormalConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    sampling: SamplingConfig = SamplingConfig()
    registration: RegistrationConfig = RegistrationConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    math: MathConfig = DEFAULT_MATH_CONFIG


# ============================================================================
# Singleton and Utilities
# ============================================================================

#: The default processing configuration instance.
DEFAULT_PROCESSING_CONFIG: Final[ProcessingConfig] = ProcessingConfig()

__all__ = [
    "NeighborConfig",
    "OutlierConfig",
    "NormalConfig",
    "ClassifierConfig",
    "SamplingConfig",
    "RegistrationConfig",
    "SegmentationConfig",
    "ProcessingConfig",
    "DEFAULT_PROCESSING_CONFIG",
]
