"""
topocore.features.buildings.walls
====================================

Wall face detection: BUILDING points whose normal is nearly
horizontal (vertical facade), clustered into linear runs.

Requires `normals` in addition to `classification`, since vertical-
vs-horizontal orientation is what separates a wall face from a roof
plane within the same BUILDING classification bucket.

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
    ClusterDetectorBase,
    ClusterFilterConfig,
    cluster_points_2d,
    convex_hull_polygon,
    extract_xyz,
)
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


class WallDetector(ClusterDetectorBase):
    """
    Detects wall faces from BUILDING points with near-horizontal
    normals.

    Parameters
    ----------
    max_normal_z
        Upper bound on ``|normal.z|`` for a point to count as
        "vertical enough" to be a wall face (0 = perfectly vertical
        wall, 1 = perfectly horizontal roof/floor).
    eps, min_points
        Clustering parameters, as in `BuildingDetector`.
    """

    category = FeatureCategory.BUILDING
    feature_type = FeatureType.WALL
    geometry_type = GeometryType.POLYGON
    classification_codes = (PointClassification.BUILDING,)
    version = "1.0"

    required_inputs = frozenset({ContextField.POINT_CLOUD, ContextField.CLASSIFICATION, ContextField.NORMALS})

    __slots__ = ("_max_normal_z",)

    def __init__(
        self,
        max_normal_z: float = 0.3,
        eps: float = 0.5,
        min_points: int = 15,
    ) -> None:
        if not 0.0 <= max_normal_z <= 1.0:
            raise DetectionError(f"max_normal_z must be in [0, 1]; got {max_normal_z}.")
        super().__init__(ClusterFilterConfig(eps=eps, min_points=min_points))
        self._max_normal_z = float(max_normal_z)

    @override
    def name(self) -> str:
        return "walls"

    @override
    def _detect(self, context: DetectionContext) -> FeatureCollection:
        normals = context.normals
        assert normals is not None

        classification = context.classification
        assert classification is not None

        cloud = context.cloud
        assert cloud is not None

        if normals.point_count != len(classification.labels):
            raise DetectionError("Classification and normals have different sizes.")

        xyz = extract_xyz(cloud)

        codes = np.array(
            [c.value for c in self.classification_codes],
            dtype=classification.labels.dtype,
        )

        mask = np.isin(classification.labels, codes)

        vertical_mask = mask & (np.abs(normals.normals[:, 2]) <= self._max_normal_z)

        candidate_idx = np.nonzero(vertical_mask)[0]

        if candidate_idx.size == 0:
            return FeatureCollection()

        clusters = cluster_points_2d(
            xyz[candidate_idx, :2],
            self._config.eps,
            self._config.min_points,
        )

        result = FeatureCollection()

        for local_id, local_idx in enumerate(clusters, start=1):
            cluster_xyz = xyz[candidate_idx[local_idx]]

            if not self._passes_filters(cluster_xyz):
                continue

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
                                ContextField.NORMALS,
                            }
                        ),
                        max_normal_z=self._max_normal_z,
                        cluster_size=int(local_idx.size),
                    ),
                    attributes=self._build_attributes(cluster_xyz),
                    source_point_indices=candidate_idx[local_idx].astype(np.int64),
                )
            )

        return result


DetectorRegistry.register(WallDetector)

__all__ = ["WallDetector"]
