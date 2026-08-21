"""
topocore.features.buildings.retaining_walls
==============================================

Retaining wall detection: near-vertical GROUND-classified structures
(as opposed to `WallDetector`, which looks at BUILDING-classified
facades). Uses the `verticality` geometric feature rather than raw
normals, since retaining walls are typically identified downstream
of PCA feature computation in the classification pipeline.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import override

import numpy as np

from topocore.features._shared import (
    cluster_points_2d,
    convex_hull_polygon,
    extract_xyz,
)
from topocore.features.base import BaseFeatureDetector
from topocore.features.detector import DetectorRegistry
from topocore.features.exceptions import DetectionError
from topocore.features.models import (
    ContextField,
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
from topocore.features.protocols import DetectionContext
from topocore.pointcloud.classification import PointClassification


class RetainingWallDetector(BaseFeatureDetector):
    """
    Detects retaining walls from GROUND points with high verticality.

    Parameters
    ----------
    min_verticality
        Minimum ``pca_features["verticality"]`` value (0-1) for a
        point to be considered part of a retaining wall face.
    eps, min_points, min_height
        Clustering / size parameters.
    """

    category = FeatureCategory.BUILDING
    feature_type = FeatureType.RETAINING_WALL
    geometry_type = GeometryType.POLYGON
    version = "1.0"
    required_inputs = frozenset(
        {
            ContextField.POINT_CLOUD,
            ContextField.CLASSIFICATION,
            ContextField.PCA_FEATURES,
        }
    )

    __slots__ = ("_eps", "_min_height", "_min_points", "_min_verticality")

    def __init__(
        self,
        min_verticality: float = 0.8,
        eps: float = 0.5,
        min_points: int = 15,
        min_height: float = 0.4,
    ) -> None:
        if not 0.0 <= min_verticality <= 1.0:
            raise DetectionError(f"min_verticality must be in [0, 1]; got {min_verticality}.")
        self._min_verticality = float(min_verticality)
        self._eps = float(eps)
        self._min_points = int(min_points)
        self._min_height = float(min_height)

    @override
    def name(self) -> str:
        return "retaining_walls"

    @override
    def _detect(self, context: DetectionContext) -> FeatureCollection:
        classification = context.classification
        pca = context.pca_features
        cloud = context.cloud

        assert classification is not None
        assert pca is not None
        assert cloud is not None

        verticality = pca["verticality"]

        if len(classification.labels) != len(verticality):
            raise DetectionError("Classification and PCA features have different sizes.")

        xyz = extract_xyz(cloud)

        ground_mask = classification.labels == PointClassification.GROUND.value

        vertical_mask = ground_mask & (verticality >= self._min_verticality)

        candidate_idx = np.nonzero(vertical_mask)[0]

        if candidate_idx.size == 0:
            return FeatureCollection()

        clusters = cluster_points_2d(
            xyz[candidate_idx, :2],
            self._eps,
            self._min_points,
        )

        result = FeatureCollection()
        local_id = 0

        # Found and fixed in PR19: confirmed reachable here (unlike
        # WallDetector) since min_height is an ACTIVE, default-on
        # filter (0.4m) -- a real cluster shorter than that mid-
        # sequence caused a gap in the surviving features' feature_id
        # (verified directly: [1, 3] instead of [1, 2] for 2 kept
        # clusters out of 3). Fixed to only increment local_id on
        # successful addition, matching the established correct
        # pattern used elsewhere (ClusterDetectorBase's own template
        # method, terrain detectors, etc.).
        for local_idx in clusters:
            cluster_xyz = xyz[candidate_idx[local_idx]]

            height = float(cluster_xyz[:, 2].max() - cluster_xyz[:, 2].min())

            if height < self._min_height:
                continue

            local_id += 1

            mean_verticality = float(np.mean(verticality[candidate_idx[local_idx]]))

            result.add(
                Feature(
                    feature_id=local_id,
                    category=self.category,
                    feature_type=self.feature_type,
                    geometry=FeatureGeometry(
                        geometry_type=self.geometry_type,
                        vertices=convex_hull_polygon(cluster_xyz),
                        closed=True,
                    ),
                    confidence=1.0,
                    metadata=self._metadata(
                        inputs_used=frozenset(
                            {
                                ContextField.POINT_CLOUD,
                                ContextField.CLASSIFICATION,
                                ContextField.PCA_FEATURES,
                            }
                        ),
                        min_verticality=self._min_verticality,
                        cluster_size=int(local_idx.size),
                        height_m=height,
                    ),
                    attributes={
                        "height_m": height,
                        "point_count": int(cluster_xyz.shape[0]),
                        "mean_verticality": mean_verticality,
                    },
                    source_point_indices=candidate_idx[local_idx].astype(np.int64),
                )
            )

        return result


DetectorRegistry.register(RetainingWallDetector)

__all__ = ["RetainingWallDetector"]
