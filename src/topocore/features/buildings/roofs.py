"""
topocore.features.buildings.roofs
====================================

Roof plane detection: BUILDING points with near-vertical normals
(complement of `WallDetector`'s near-horizontal-normal criterion),
grouped by similar normal orientation into roof planes, each
triangulated into a MESH.

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
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError

from topocore.features._shared import cluster_points_2d, extract_xyz
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


class RoofDetector(BaseFeatureDetector):
    """
    Detect roof planes from BUILDING-classified points.

    Candidate roof points are selected from their normal orientation,
    grouped by similar normal direction, spatially clustered, and
    finally triangulated into 3D mesh geometries.

    Parameters
    ----------
    min_normal_z
        Lower bound on ``normal.z`` for a point to count as a roof
        surface rather than a wall face.
    orientation_angle_deg
        Maximum angular difference, in degrees, used when grouping
        points with similar normal orientation.
    eps
        Maximum XY distance between neighboring points during
        spatial clustering.
    min_points
        Minimum number of points required for a spatial cluster.
    """

    category = FeatureCategory.BUILDING
    feature_type = FeatureType.ROOF
    geometry_type = GeometryType.MESH
    version = "1.0"

    required_inputs = frozenset(
        {
            ContextField.POINT_CLOUD,
            ContextField.CLASSIFICATION,
            ContextField.NORMALS,
        }
    )

    __slots__ = (
        "_min_normal_z",
        "_orientation_angle_deg",
        "_eps",
        "_min_points",
    )

    def __init__(
        self,
        min_normal_z: float = 0.5,
        orientation_angle_deg: float = 15.0,
        eps: float = 0.7,
        min_points: int = 20,
    ) -> None:
        if not 0.0 <= min_normal_z <= 1.0:
            raise DetectionError(f"min_normal_z must be in [0, 1]; got {min_normal_z}.")

        if not 0.0 < orientation_angle_deg < 90.0:
            raise DetectionError(f"orientation_angle_deg must be in (0, 90); got {orientation_angle_deg}.")

        if eps <= 0.0:
            raise DetectionError(f"eps must be positive; got {eps}.")

        if min_points < 1:
            raise DetectionError(f"min_points must be >= 1; got {min_points}.")

        self._min_normal_z = float(min_normal_z)
        self._orientation_angle_deg = float(orientation_angle_deg)
        self._eps = float(eps)
        self._min_points = int(min_points)

    @override
    def name(self) -> str:
        return "roofs"

    @override
    def _detect(
        self,
        context: DetectionContext,
    ) -> FeatureCollection:
        cloud = context.cloud
        classification = context.classification
        normal_result = context.normals

        assert cloud is not None
        assert classification is not None
        assert normal_result is not None

        normals = normal_result.normals

        point_count = cloud.point_count

        if classification.labels.shape[0] != point_count:
            raise DetectionError(
                f"Classification size does not match point cloud: {classification.labels.shape[0]} != {point_count}."
            )

        if normals.shape[0] != point_count:
            raise DetectionError(f"Normals size does not match point cloud: {normals.shape[0]} != {point_count}.")

        xyz = extract_xyz(cloud)

        if xyz.shape[0] != point_count:
            raise DetectionError(f"Extracted XYZ size does not match point cloud: {xyz.shape[0]} != {point_count}.")

        building_mask = classification.labels == PointClassification.BUILDING.value

        roof_mask = building_mask & (normals[:, 2] >= self._min_normal_z)

        candidate_idx = np.flatnonzero(roof_mask)

        if candidate_idx.size == 0:
            return FeatureCollection()

        # Only normals need to be materialized here because orientation
        # grouping operates directly on the candidate subset. XYZ
        # coordinates remain in the original array and are accessed
        # through source indices to avoid keeping another full
        # candidate coordinate array.
        candidate_normals = normals[candidate_idx]

        orientation_groups = self._group_by_orientation(
            candidate_normals,
        )

        result = FeatureCollection()
        local_id = 0

        for group_idx in orientation_groups:
            if group_idx.size == 0:
                continue

            # Convert orientation-group-local indices back into source
            # point-cloud indices.
            group_source_idx = candidate_idx[group_idx]

            spatial_clusters = cluster_points_2d(
                xyz[group_source_idx, :2],
                self._eps,
                self._min_points,
            )

            for cluster_local_idx in spatial_clusters:
                if cluster_local_idx.size == 0:
                    continue

                source_idx = group_source_idx[cluster_local_idx]
                cluster_xyz = xyz[source_idx]

                geometry = self._triangulate(
                    cluster_xyz,
                )

                if geometry is None:
                    continue

                cluster_normals = normals[source_idx]

                mean_normal = np.asarray(
                    cluster_normals.mean(axis=0),
                    dtype=np.float64,
                )

                mean_normal_norm = float(np.linalg.norm(mean_normal))

                if mean_normal_norm > 0.0:
                    mean_normal = mean_normal / mean_normal_norm

                mean_normal_z = float(np.mean(cluster_normals[:, 2]))

                local_id += 1

                result.add(
                    Feature(
                        feature_id=local_id,
                        category=self.category,
                        feature_type=self.feature_type,
                        geometry=geometry,
                        confidence=1.0,
                        metadata=self._metadata(
                            inputs_used=frozenset(
                                {
                                    ContextField.POINT_CLOUD,
                                    ContextField.CLASSIFICATION,
                                    ContextField.NORMALS,
                                }
                            ),
                            min_normal_z=self._min_normal_z,
                            orientation_angle_deg=self._orientation_angle_deg,
                            eps=self._eps,
                            min_points=self._min_points,
                            cluster_size=int(source_idx.size),
                        ),
                        attributes={
                            "point_count": int(source_idx.size),
                            "mean_normal_x": float(mean_normal[0]),
                            "mean_normal_y": float(mean_normal[1]),
                            "mean_normal_z": mean_normal_z,
                        },
                        source_point_indices=np.asarray(
                            source_idx,
                            dtype=np.int64,
                        ),
                    )
                )

        return result

    def _group_by_orientation(
        self,
        normals: NDArray[np.float64],
    ) -> list[NDArray[np.int64]]:
        """
        Group candidate normals by similar orientation.

        A greedy representative-based grouping is used. Each group
        maintains the accumulated normal direction of all points
        assigned to it, so its representative evolves as points are
        added rather than remaining fixed at the first normal.

        Parameters
        ----------
        normals
            Candidate unit normals with shape ``(n, 3)``.

        Returns
        -------
        list[NDArray[np.int64]]
            Candidate-local point indices for each orientation group.
        """
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise DetectionError(f"Normals for orientation grouping must have shape (n, 3); got {normals.shape}.")

        if normals.shape[0] == 0:
            return []

        threshold = float(
            np.cos(
                np.radians(
                    self._orientation_angle_deg,
                )
            )
        )

        point_count = normals.shape[0]

        assigned = np.full(
            point_count,
            -1,
            dtype=np.int64,
        )

        group_sums: list[NDArray[np.float64]] = []
        group_representatives: list[NDArray[np.float64]] = []

        for point_idx in range(point_count):
            normal = normals[point_idx]

            normal_norm = float(np.linalg.norm(normal))

            if normal_norm <= np.finfo(np.float64).eps:
                continue

            unit_normal = np.asarray(
                normal / normal_norm,
                dtype=np.float64,
            )

            best_group = -1
            best_similarity = threshold

            for group_idx, representative in enumerate(group_representatives):
                similarity = float(
                    np.dot(
                        unit_normal,
                        representative,
                    )
                )

                if similarity >= best_similarity:
                    best_similarity = similarity
                    best_group = group_idx

            if best_group < 0:
                group_sums.append(unit_normal.copy())
                group_representatives.append(unit_normal.copy())

                best_group = len(group_representatives) - 1

            else:
                group_sum = group_sums[best_group] + unit_normal

                group_sums[best_group] = group_sum

                group_norm = float(np.linalg.norm(group_sum))

                if group_norm > np.finfo(np.float64).eps:
                    group_representatives[best_group] = np.asarray(
                        group_sum / group_norm,
                        dtype=np.float64,
                    )

            assigned[point_idx] = best_group

        return [
            np.asarray(
                np.flatnonzero(assigned == group_idx),
                dtype=np.int64,
            )
            for group_idx in range(len(group_representatives))
        ]

    @classmethod
    def _triangulate(
        cls,
        cluster_xyz: NDArray[np.float64],
    ) -> FeatureGeometry | None:
        """
        Triangulate a roof-plane cluster in XY.

        Parameters
        ----------
        cluster_xyz
            Roof-plane points with shape ``(n, 3)``.

        Returns
        -------
        FeatureGeometry | None
            Triangular mesh, or ``None`` when the cluster cannot
            produce a valid 2D Delaunay triangulation.
        """
        if cluster_xyz.ndim != 2 or cluster_xyz.shape[1] != 3:
            raise DetectionError(f"Roof cluster coordinates must have shape (n, 3); got {cluster_xyz.shape}.")

        if cluster_xyz.shape[0] < 3:
            return None

        xy = cluster_xyz[:, :2]

        if (
            np.unique(
                xy,
                axis=0,
            ).shape[0]
            < 3
        ):
            return None

        try:
            triangulation = Delaunay(xy)
        except QhullError:
            return None

        faces = np.asarray(
            triangulation.simplices,
            dtype=np.int64,
        )

        if faces.size == 0:
            return None

        return FeatureGeometry(
            geometry_type=cls.geometry_type,
            vertices=cluster_xyz,
            faces=faces,
        )


DetectorRegistry.register(RoofDetector)

__all__ = ["RoofDetector"]
