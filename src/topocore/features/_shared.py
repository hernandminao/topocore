"""
topocore.features._shared
============================

Shared, private utilities reused across detector subpackages:
point cloud coordinate extraction, connected-components clustering,
convex-hull footprints, and a generic cluster-based detector base
class that most point-driven detectors (buildings, vegetation,
utilities, infrastructure) build on.

Not part of the public API — detector modules import from here, but
`topocore.features.__init__` does not re-export any of it.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, cKDTree

from topocore.core.types import IntArray1D
from topocore.features.base import BaseFeatureDetector
from topocore.features.models import (
    ContextField,
    Feature,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
from topocore.features.protocols import DetectionContext
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud


def extract_xyz(cloud: PointCloud) -> NDArray[np.float64]:
    """
    Extract XYZ coordinates from a `PointCloud` as one ``(n, 3)`` array.

    Mirrors the chunked-attribute-access pattern already used in
    ``processing.classification.ml.MachineLearningClassifier``.
    """
    from topocore.pointcloud.attributes import PointAttribute

    xs = np.concatenate([chunk[PointAttribute.X] for chunk in cloud]).astype(np.float64)

    ys = np.concatenate([chunk[PointAttribute.Y] for chunk in cloud]).astype(np.float64)

    zs = np.concatenate([chunk[PointAttribute.Z] for chunk in cloud]).astype(np.float64)

    return np.column_stack([xs, ys, zs])


def cluster_points_2d(
    xy: NDArray[np.float64],
    eps: float,
    min_points: int,
) -> list[NDArray[np.int64]]:
    """
    Group points into connected components using a 2D radius graph.

    This is single-linkage connected-components clustering at radius
    `eps`: every point within `eps` of another point in the same
    cluster joins it.

    Parameters
    ----------
    xy
        ``(n, 2)`` point coordinates.
    eps
        Maximum distance between points for them to be linked.
    min_points
        Minimum cluster size to keep.

    Returns
    -------
    list[NDArray[np.int64]]
        Each array holds indices into `xy` for one cluster.
    """
    n = xy.shape[0]

    if n == 0:
        return []

    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")

    parent = np.arange(n)

    def find(x: int) -> int:
        root = x

        while parent[root] != root:
            root = parent[root]

        while parent[x] != root:
            parent[x], x = root, parent[x]

        return int(root)

    for a, b in pairs.tolist():
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}

    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return [np.asarray(indices, dtype=np.int64) for indices in groups.values() if len(indices) >= min_points]


def convex_hull_polygon(
    points_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Build a 2D convex-hull footprint from a point cluster.

    Z is set to the cluster's median elevation, providing a robust
    representative elevation for the footprint polygon.

    For clusters with fewer than three distinct XY positions, a
    minimal fallback triangle is generated.
    """
    xy = points_xyz[:, :2]
    z = float(np.median(points_xyz[:, 2]))

    unique_xy = np.unique(xy, axis=0)

    if unique_xy.shape[0] < 3:
        if unique_xy.shape[0] > 0:
            base = unique_xy[0]
        else:
            base = np.asarray([0.0, 0.0], dtype=np.float64)

        offsets = np.asarray(
            [
                [0.0, 0.0],
                [1e-3, 0.0],
                [0.0, 1e-3],
            ],
            dtype=np.float64,
        )

        hull_xy = base + offsets

    else:
        hull = ConvexHull(unique_xy)
        hull_xy = unique_xy[hull.vertices]

    return np.column_stack(
        [
            hull_xy,
            np.full(hull_xy.shape[0], z, dtype=np.float64),
        ]
    )


def horizontal_extent(
    points_xyz: NDArray[np.float64],
) -> float:
    """
    Return the diagonal of the cluster's XY bounding box.
    """
    xy = points_xyz[:, :2]

    return float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))


def elongation_ratio(
    points_xyz: NDArray[np.float64],
) -> float:
    """
    Return the ratio of the two principal XY spread directions.

    Returns
    -------
    float
        Ratio >= 1.0. Values near 1.0 indicate compact clusters;
        larger values indicate increasingly elongated clusters.
    """
    xy = points_xyz[:, :2]
    centered = xy - xy.mean(axis=0)

    if centered.shape[0] < 2:
        return 1.0

    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)

    return float(np.sqrt(eigenvalues.max() / eigenvalues.min()))


@dataclass(slots=True)
class ClusterFilterConfig:
    """
    Geometric filters applied to each candidate cluster after
    connected-components grouping.

    Any optional bound left as ``None`` is not enforced.
    """

    eps: float
    min_points: int
    min_height: float | None = None
    max_height: float | None = None
    min_horizontal_extent: float | None = None
    max_horizontal_extent: float | None = None
    min_elongation: float | None = None
    max_elongation: float | None = None


class ClusterDetectorBase(BaseFeatureDetector):
    """
    Shared pipeline for point-driven feature detectors.

    The pipeline:

    1. Extract XYZ coordinates.
    2. Filter points by classification.
    3. Cluster candidate points spatially in XY.
    4. Apply geometric filters.
    5. Build geometry and metadata.
    6. Emit one `Feature` per accepted cluster.

    Subclasses normally define `classification_codes`,
    `feature_type`, and `geometry_type` as class variables.

    Detectors requiring classification configuration per instance
    may override `_classification_codes()` instead of mutating the
    class-level `classification_codes`.
    """

    classification_codes: ClassVar[tuple[PointClassification, ...]] = ()

    feature_type: ClassVar[FeatureType]

    geometry_type: ClassVar[GeometryType] = GeometryType.POINT

    point_strategy: ClassVar[Literal["centroid", "min_z", "max_z"]] = "centroid"

    required_inputs = frozenset(
        {
            ContextField.POINT_CLOUD,
            ContextField.CLASSIFICATION,
        }
    )

    __slots__ = ("_config",)

    def __init__(
        self,
        config: ClusterFilterConfig,
    ) -> None:
        self._config = config

    def _classification_codes(
        self,
    ) -> tuple[PointClassification, ...]:
        """
        Return classification codes accepted by this detector.

        The default implementation returns the detector's class-level
        `classification_codes`.

        Subclasses requiring per-instance classification selection
        should override this method rather than assigning to the
        `ClassVar` through an instance.
        """
        return self.classification_codes

    def _detect(
        self,
        context: DetectionContext,
    ) -> FeatureCollection:
        """
        Detect features from classified point clusters.
        """
        assert context.cloud is not None
        assert context.classification is not None

        xyz = extract_xyz(context.cloud)

        classification_codes = self._classification_codes()

        codes = np.asarray(
            [code.value for code in classification_codes],
            dtype=np.uint8,
        )

        candidate_mask = np.isin(
            context.classification.labels,
            codes,
        )

        candidate_idx = np.flatnonzero(candidate_mask)

        if candidate_idx.size == 0:
            return FeatureCollection()

        candidate_xyz = xyz[candidate_idx]

        clusters = cluster_points_2d(
            candidate_xyz[:, :2],
            eps=self._config.eps,
            min_points=self._config.min_points,
        )

        result = FeatureCollection()
        local_id = 0

        for local_cluster in clusters:
            # Convert cluster-local indices back to original
            # point-cloud indices.
            cluster_idx = candidate_idx[local_cluster]

            # Index XYZ directly from the original array. This avoids
            # copying candidate_xyz[local_cluster] after candidate_xyz
            # has already been materialized.
            cluster_xyz = xyz[cluster_idx]

            if not self._passes_filters(cluster_xyz):
                continue

            local_id += 1

            result.add(
                self._build_feature(
                    local_id=local_id,
                    cluster_xyz=cluster_xyz,
                    source_indices=cluster_idx.astype(
                        np.int64,
                        copy=False,
                    ),
                )
            )

        return result

    def _build_feature(
        self,
        *,
        local_id: int,
        cluster_xyz: NDArray[np.float64],
        source_indices: IntArray1D,
    ) -> Feature:
        """
        Build a feature from a validated point cluster.

        Geometry construction is delegated to `_build_geometry()`,
        allowing subclasses to override geometry creation while
        retaining the common metadata and provenance pipeline.
        """
        geometry = self._build_geometry(cluster_xyz)

        centroid = np.asarray(
            cluster_xyz.mean(axis=0),
            dtype=np.float64,
        )

        bbox_min = cluster_xyz.min(axis=0)
        bbox_max = cluster_xyz.max(axis=0)

        return Feature(
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
                    }
                ),
                cluster_size=int(cluster_xyz.shape[0]),
                eps=self._config.eps,
                centroid=(
                    float(centroid[0]),
                    float(centroid[1]),
                    float(centroid[2]),
                ),
                bounds=(
                    float(bbox_min[0]),
                    float(bbox_min[1]),
                    float(bbox_min[2]),
                    float(bbox_max[0]),
                    float(bbox_max[1]),
                    float(bbox_max[2]),
                ),
            ),
            attributes=self._build_attributes(cluster_xyz),
            source_point_indices=source_indices,
        )

    def _passes_filters(
        self,
        cluster_xyz: NDArray[np.float64],
    ) -> bool:
        """
        Validate a cluster against configured geometric constraints.

        Expensive quantities are computed only when their
        corresponding filters are enabled.
        """
        config = self._config

        if cluster_xyz.shape[0] < config.min_points:
            return False

        if config.min_height is not None or config.max_height is not None:
            height = float(cluster_xyz[:, 2].max() - cluster_xyz[:, 2].min())

            if config.min_height is not None and height < config.min_height:
                return False

            if config.max_height is not None and height > config.max_height:
                return False

        if config.min_horizontal_extent is not None or config.max_horizontal_extent is not None:
            extent = horizontal_extent(cluster_xyz)

            if config.min_horizontal_extent is not None and extent < config.min_horizontal_extent:
                return False

            if config.max_horizontal_extent is not None and extent > config.max_horizontal_extent:
                return False

        if config.min_elongation is not None or config.max_elongation is not None:
            ratio = elongation_ratio(cluster_xyz)

            if config.min_elongation is not None and ratio < config.min_elongation:
                return False

            if config.max_elongation is not None and ratio > config.max_elongation:
                return False

        return True

    def _build_geometry(
        self,
        cluster_xyz: NDArray[np.float64],
    ) -> FeatureGeometry:
        """
        Build geometry for an accepted cluster.

        POLYGON detectors use an XY convex-hull footprint with
        representative Z.

        POINT detectors use one representative XYZ coordinate.

        Subclasses requiring POLYLINE, MESH, or specialized geometry
        should override this method.
        """
        if self.geometry_type == GeometryType.POLYGON:
            return FeatureGeometry(
                geometry_type=GeometryType.POLYGON,
                vertices=convex_hull_polygon(cluster_xyz),
                closed=True,
            )

        if self.geometry_type == GeometryType.POINT:
            point = self._representative_point(cluster_xyz)

            return FeatureGeometry(
                geometry_type=GeometryType.POINT,
                vertices=point[None, :],
            )

        raise NotImplementedError(
            f"{type(self).__name__} does not define a geometry builder for "
            f"{self.geometry_type.value}; override `_build_geometry`."
        )

    def _representative_point(
        self,
        cluster_xyz: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Return the representative XYZ point for a cluster.
        """
        if self.point_strategy == "min_z":
            point = cluster_xyz[int(np.argmin(cluster_xyz[:, 2]))]

        elif self.point_strategy == "max_z":
            point = cluster_xyz[int(np.argmax(cluster_xyz[:, 2]))]

        else:
            point = cluster_xyz.mean(axis=0)

        return np.asarray(
            point,
            dtype=np.float64,
        )

    def _build_attributes(
        self,
        cluster_xyz: NDArray[np.float64],
    ) -> dict[str, object]:
        """
        Build common semantic attributes for a point cluster.
        """
        return {
            "point_count": int(cluster_xyz.shape[0]),
            "height_m": float(cluster_xyz[:, 2].max() - cluster_xyz[:, 2].min()),
            "horizontal_extent_m": horizontal_extent(cluster_xyz),
        }


__all__ = [
    "extract_xyz",
    "cluster_points_2d",
    "convex_hull_polygon",
    "horizontal_extent",
    "elongation_ratio",
    "ClusterFilterConfig",
    "ClusterDetectorBase",
]
