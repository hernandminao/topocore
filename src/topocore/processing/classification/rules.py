"""
topocore.processing.classification.rules
========================================

Rule-based classification for point clouds.

This module implements rule-based classification using geometric
and radiometric heuristics. Rules are defined as a set of conditions
that assign ASPRS classification codes.

Rules are designed to be configurable and extensible.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import ProcessingError
from topocore.processing.features import PCAFeatures
from topocore.processing.filters import RadiusOutlierFilter
from topocore.processing.ground import GroundManager
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import BoolArray1D, FloatArray1D, IntArray1D

from .base import ClassificationResult
from .base import RuleBasedClassifier as _RuleBasedClassifier


@dataclass(slots=True, frozen=True)
class _RuleFeatures:
    """
    Features required by rule-based classification.
    """

    height_above_ground: FloatArray1D
    curvature: FloatArray1D
    planarity: FloatArray1D
    linearity: FloatArray1D
    sphericity: FloatArray1D
    verticality: FloatArray1D
    anisotropy: FloatArray1D
    omnivariance: FloatArray1D
    density: FloatArray1D
    not_noise: BoolArray1D


class RuleBasedClassifier(_RuleBasedClassifier):
    """
    Rule-based point cloud classifier.

    Uses hand-crafted rules to assign ASPRS classification codes:
    - Ground: points near the ground surface
    - Low Vegetation: low height above ground, irregular local geometry
    - Medium Vegetation: medium height above ground, irregular local geometry
    - High Vegetation: high height above ground, irregular local geometry
    - Building: planar, non-spherical elevated points
    - Noise: isolated points with few neighbors

    Parameters
    ----------
    ground_cell_size
        Cell size for ground estimation.
    ground_max_height
        Maximum height above ground for ground classification.
    low_vegetation_max_height
        Maximum height for low vegetation.
    high_vegetation_min_height
        Minimum height for high vegetation.
    curvature_threshold
        Minimum curvature required for vegetation classification.
    noise_radius
        Radius for noise detection.
    noise_min_neighbors
        Minimum neighbors for noise detection.
    vegetation_max_anisotropy
        Maximum anisotropy allowed for vegetation classification.
    vegetation_min_omnivariance
        Minimum omnivariance required for vegetation classification.
    """

    __slots__ = (
        "_ground_cell_size",
        "_ground_max_height",
        "_low_vegetation_max_height",
        "_high_vegetation_min_height",
        "_curvature_threshold",
        "_noise_radius",
        "_noise_min_neighbors",
        "_pca_neighbors",
        "_vegetation_min_height",
        "_vegetation_planarity_threshold",
        "_vegetation_linearity_threshold",
        "_vegetation_sphericity_threshold",
        "_vegetation_max_anisotropy",
        "_vegetation_min_omnivariance",
        "_building_planarity_threshold",
        "_building_sphericity_threshold",
        "_building_min_verticality",
        "_building_max_verticality",
        "_max_building_height",
        "_ground_manager",
        "_noise_filter",
        "_pca_features",
        "_rules_description",
    )

    def __init__(
        self,
        ground_cell_size: float = 1.0,
        ground_max_height: float = 0.2,
        low_vegetation_max_height: float = 0.5,
        high_vegetation_min_height: float = 2.0,
        curvature_threshold: float = 0.02,
        noise_radius: float = 1.0,
        noise_min_neighbors: int = 4,
        pca_neighbors: int = 10,
        vegetation_min_height: float = 0.1,
        vegetation_planarity_threshold: float = 0.6,
        vegetation_linearity_threshold: float = 0.8,
        vegetation_sphericity_threshold: float = 0.2,
        vegetation_max_anisotropy: float = 0.8,
        vegetation_min_omnivariance: float = 0.01,
        building_planarity_threshold: float = 0.5,
        building_sphericity_threshold: float = 0.3,
        building_min_verticality: float = 0.3,
        building_max_verticality: float = 0.7,
        max_building_height: float = 50.0,
    ) -> None:
        if ground_cell_size <= 0:
            raise ValueError("ground_cell_size must be positive.")

        if ground_max_height < 0:
            raise ValueError("ground_max_height must be non-negative.")

        if low_vegetation_max_height < 0:
            raise ValueError("low_vegetation_max_height must be non-negative.")

        if high_vegetation_min_height <= low_vegetation_max_height:
            raise ValueError("high_vegetation_min_height must be greater than low_vegetation_max_height.")

        if curvature_threshold < 0:
            raise ValueError("curvature_threshold must be non-negative.")

        if noise_radius <= 0:
            raise ValueError("noise_radius must be positive.")

        if noise_min_neighbors < 1:
            raise ValueError("noise_min_neighbors must be at least 1.")

        if pca_neighbors < 3:
            raise ValueError("pca_neighbors must be at least 3.")

        if not 0.0 <= building_planarity_threshold <= 1.0:
            raise ValueError("building_planarity_threshold must be in the range [0, 1].")

        if not 0.0 <= building_sphericity_threshold <= 1.0:
            raise ValueError("building_sphericity_threshold must be in the range [0, 1].")

        if not 0.0 <= building_min_verticality <= 1.0:
            raise ValueError("building_min_verticality must be between 0 and 1.")

        if not 0.0 <= building_max_verticality <= 1.0:
            raise ValueError("building_max_verticality must be between 0 and 1.")

        if building_min_verticality >= building_max_verticality:
            raise ValueError("building_min_verticality must be less than building_max_verticality.")

        if max_building_height <= 0:
            raise ValueError("max_building_height must be positive.")

        if vegetation_min_height < 0:
            raise ValueError("vegetation_min_height must be non-negative.")

        if not 0.0 <= vegetation_planarity_threshold <= 1.0:
            raise ValueError("vegetation_planarity_threshold must be between 0 and 1.")

        if not 0.0 <= vegetation_linearity_threshold <= 1.0:
            raise ValueError("vegetation_linearity_threshold must be between 0 and 1.")

        if not 0.0 <= vegetation_sphericity_threshold <= 1.0:
            raise ValueError("vegetation_sphericity_threshold must be between 0 and 1.")

        if not 0.0 <= vegetation_max_anisotropy <= 1.0:
            raise ValueError("vegetation_max_anisotropy must be between 0 and 1.")

        if vegetation_min_omnivariance < 0:
            raise ValueError("vegetation_min_omnivariance must be non-negative.")

        if vegetation_min_height > low_vegetation_max_height:
            raise ValueError("vegetation_min_height must be <= low_vegetation_max_height.")

        self._ground_cell_size = ground_cell_size
        self._ground_max_height = ground_max_height
        self._low_vegetation_max_height = low_vegetation_max_height
        self._high_vegetation_min_height = high_vegetation_min_height
        self._curvature_threshold = curvature_threshold
        self._noise_radius = noise_radius
        self._noise_min_neighbors = noise_min_neighbors
        self._pca_neighbors = pca_neighbors
        self._building_planarity_threshold = building_planarity_threshold
        self._building_sphericity_threshold = building_sphericity_threshold
        self._building_min_verticality = building_min_verticality
        self._building_max_verticality = building_max_verticality
        self._max_building_height = max_building_height
        self._vegetation_min_height = vegetation_min_height
        self._vegetation_planarity_threshold = vegetation_planarity_threshold
        self._vegetation_linearity_threshold = vegetation_linearity_threshold
        self._vegetation_sphericity_threshold = vegetation_sphericity_threshold
        self._vegetation_max_anisotropy = vegetation_max_anisotropy
        self._vegetation_min_omnivariance = vegetation_min_omnivariance

        self._ground_manager = GroundManager(
            method="grid",
            cell_size=ground_cell_size,
        )

        self._noise_filter = RadiusOutlierFilter(
            radius=noise_radius,
            min_neighbors=noise_min_neighbors,
            include_self=True,
        )

        self._pca_features = PCAFeatures(k=pca_neighbors)

        # Pre-cache rules description
        self._rules_description = {
            "noise": (f"Points with fewer than {self._noise_min_neighbors} neighbors within {self._noise_radius}m"),
            "ground": f"Points with height above ground <= {self._ground_max_height}m",
            "building": (
                f"Planar points with planarity >= {self._building_planarity_threshold} "
                f"and sphericity <= {self._building_sphericity_threshold} "
                f"and height <= {self._max_building_height}m"
            ),
            "vegetation": (
                "Points satisfying vegetation geometry rules: "
                f"height >= {self._vegetation_min_height}m, "
                f"curvature >= {self._curvature_threshold}, "
                f"planarity <= {self._vegetation_planarity_threshold}, "
                f"linearity <= {self._vegetation_linearity_threshold}, "
                f"sphericity >= {self._vegetation_sphericity_threshold}, "
                f"anisotropy <= {self._vegetation_max_anisotropy}, "
                f"omnivariance >= {self._vegetation_min_omnivariance}"
            ),
        }

    @property
    def ground_threshold(self) -> float:
        """Return the ground height threshold."""
        return self._ground_max_height

    @property
    def curvature_threshold(self) -> float:
        """Return the curvature threshold."""
        return self._curvature_threshold

    @property
    def building_threshold(self) -> float:
        """Return the building planarity threshold."""
        return self._building_planarity_threshold

    @property
    def vegetation_threshold(self) -> float:
        """Return the vegetation planarity threshold."""
        return self._vegetation_planarity_threshold

    @override
    def classify(self, cloud: PointCloud) -> ClassificationResult:
        """
        Classify point cloud using geometric rules.
        """
        if cloud.is_empty:
            raise ProcessingError("Cannot classify an empty point cloud.")

        features = self._compute_features(cloud)
        labels = self._apply_rules(features)

        return ClassificationResult(labels=labels, cloud=cloud)

    def _compute_features(self, cloud: PointCloud) -> _RuleFeatures:
        """
        Compute all geometric features required by the rule-based classifier.

        A single NeighborhoodManager instance is reused by every processing
        stage (PCA, density and noise filtering) to avoid rebuilding the KDTree.
        """
        manager = NeighborhoodManager.from_point_cloud(cloud)

        point_count = cloud.point_count

        z = np.concatenate(
            [chunk[PointAttribute.Z] for chunk in cloud],
        ).astype(np.float64, copy=False)

        ground = self._ground_manager.estimate_elevation(cloud)
        if ground.shape[0] != point_count:
            raise ProcessingError("Ground model returned an invalid size.")

        pca = self._pca_features.compute(
            cloud,
            manager=manager,
        )

        required_pca = (
            "surface_variation",
            "planarity",
            "linearity",
            "sphericity",
            "verticality",
            "anisotropy",
            "omnivariance",
        )

        missing = [name for name in required_pca if name not in pca]

        if missing:
            raise ProcessingError("Missing PCA features: " + ", ".join(missing))

        not_noise = self._noise_filter.mask(
            cloud,
            manager=manager,
        )

        if np.all(~not_noise):
            raise ProcessingError("Noise filter rejected every point.")

        height_above_ground = np.maximum(
            z - ground,
            0.0,
        )

        density = np.empty(
            point_count,
            dtype=np.float64,
        )

        for i in range(point_count):
            density[i] = manager.local_density(
                i,
                radius=self._noise_radius,
            )

        features = _RuleFeatures(
            height_above_ground=height_above_ground,
            curvature=pca["surface_variation"],
            planarity=pca["planarity"],
            linearity=pca["linearity"],
            sphericity=pca["sphericity"],
            verticality=pca["verticality"],
            anisotropy=pca["anisotropy"],
            omnivariance=pca["omnivariance"],
            density=density,
            not_noise=not_noise,
        )

        self._validate_features(
            features,
            point_count,
        )

        return features

    def _validate_features(
        self,
        features: _RuleFeatures,
        expected_size: int,
    ) -> None:
        """
        Validate computed feature arrays.
        """
        arrays = {
            "height_above_ground": features.height_above_ground,
            "curvature": features.curvature,
            "planarity": features.planarity,
            "linearity": features.linearity,
            "sphericity": features.sphericity,
            "verticality": features.verticality,
            "anisotropy": features.anisotropy,
            "omnivariance": features.omnivariance,
            "density": features.density,
        }

        for name, array in arrays.items():
            if array.ndim != 1:
                raise ProcessingError(f"Feature '{name}' must be one-dimensional.")

            if array.shape[0] != expected_size:
                raise ProcessingError(f"Feature '{name}' has invalid size.")

            if not np.isfinite(array).all():
                raise ProcessingError(f"Feature '{name}' contains NaN or Inf.")

        if features.not_noise.shape[0] != expected_size:
            raise ProcessingError("Noise mask has invalid size.")

        if features.not_noise.dtype != np.bool_:
            raise ProcessingError("Noise mask must be boolean.")

    def _apply_rules(self, features: _RuleFeatures) -> IntArray1D:
        """
        Apply ASPRS classification rules.

        Classification priority:
        1. Noise
        2. Ground
        3. Buildings
        4. Vegetation
        """
        n_points = features.height_above_ground.shape[0]

        labels = np.full(
            n_points,
            PointClassification.UNCLASSIFIED,
            dtype=np.int64,
        )

        valid = features.not_noise
        noise = ~valid
        labels[noise] = PointClassification.HIGH_NOISE

        # Reusable ground mask
        ground = valid & (features.height_above_ground <= self._ground_max_height)
        labels[ground] = PointClassification.GROUND

        # Reusable non-ground valid points mask
        nonground = valid & ~ground

        building = self._classify_buildings(
            height_above_ground=features.height_above_ground,
            planarity=features.planarity,
            sphericity=features.sphericity,
            verticality=features.verticality,
            nonground=nonground,
            labels=labels,
        )

        self._classify_vegetation(
            height_above_ground=features.height_above_ground,
            curvature=features.curvature,
            planarity=features.planarity,
            linearity=features.linearity,
            sphericity=features.sphericity,
            anisotropy=features.anisotropy,
            omnivariance=features.omnivariance,
            nonground=nonground,
            building=building,
            labels=labels,
        )

        # Final result validation
        if labels.shape[0] != n_points:
            raise ProcessingError("Classification produced an invalid number of labels.")

        if not np.issubdtype(labels.dtype, np.integer):
            raise ProcessingError("Labels must be integers.")

        if np.all(labels == PointClassification.UNCLASSIFIED):
            raise ProcessingError("Classification failed: no rule matched.")

        return labels

    def _classify_buildings(
        self,
        height_above_ground: FloatArray1D,
        planarity: FloatArray1D,
        sphericity: FloatArray1D,
        verticality: FloatArray1D,
        nonground: BoolArray1D,
        labels: IntArray1D,
    ) -> BoolArray1D:
        """
        Classify building points using geometric features.
        """
        # Buildings are typically planar, non-spherical, and have vertical faces or horizontal roofs.
        # Verticality helps identify walls.
        building = (
            nonground
            & (planarity >= self._building_planarity_threshold)
            & (sphericity <= self._building_sphericity_threshold)
            & (height_above_ground <= self._max_building_height)
            & (
                (verticality <= self._building_min_verticality) | (verticality >= self._building_max_verticality)
            )  # Walls or Roofs
        )

        labels[building] = PointClassification.BUILDING
        return building

    def _classify_vegetation(
        self,
        height_above_ground: FloatArray1D,
        curvature: FloatArray1D,
        planarity: FloatArray1D,
        linearity: FloatArray1D,
        sphericity: FloatArray1D,
        anisotropy: FloatArray1D,
        omnivariance: FloatArray1D,
        nonground: BoolArray1D,
        building: BoolArray1D,
        labels: IntArray1D,
    ) -> BoolArray1D:
        """
        Classify vegetation using geometric features.
        """
        # Vegetation is characterized by high omnivariance (disordered),
        # low anisotropy (non-directional), and high curvature/sphericity.
        vegetation = (
            nonground
            & (~building)
            & (height_above_ground >= self._vegetation_min_height)
            & (curvature >= self._curvature_threshold)
            & (planarity <= self._vegetation_planarity_threshold)
            & (linearity <= self._vegetation_linearity_threshold)
            & (sphericity >= self._vegetation_sphericity_threshold)
            & (anisotropy <= self._vegetation_max_anisotropy)
            & (omnivariance >= self._vegetation_min_omnivariance)
        )

        low = vegetation & (height_above_ground <= self._low_vegetation_max_height)
        labels[low] = PointClassification.LOW_VEGETATION

        medium = (
            vegetation
            & (height_above_ground > self._low_vegetation_max_height)
            & (height_above_ground < self._high_vegetation_min_height)
        )
        labels[medium] = PointClassification.MEDIUM_VEGETATION

        high = vegetation & (height_above_ground >= self._high_vegetation_min_height)
        labels[high] = PointClassification.HIGH_VEGETATION

        return vegetation

    @override
    def name(self) -> str:
        return "rule_based"

    @override
    def supports_confidence(self) -> bool:
        return False

    @override
    def rules(self) -> dict[str, str]:
        """Return the pre-cached rules description."""
        return self._rules_description


__all__ = ["RuleBasedClassifier"]
