"""
topocore.processing.segmentation.specific
=========================================

Specialized segmenters for specific object types.

This module implements specialized segmenters for common object types
in point clouds:
- Trees: segmentation of individual trees
- Buildings: segmentation of building structures
- Poles: segmentation of utility poles and lamp posts
- Powerlines: segmentation of powerline conductors

These segmenters are built on top of the general segmentation
algorithms but incorporate domain-specific heuristics.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import cast, override

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import build_cloud_from_mask, flatten_attributes
from topocore.processing.exceptions import SegmentationError
from topocore.processing.features import PCAFeatures
from topocore.processing.ground import GroundManager
from topocore.processing.neighbors import NeighborhoodManager

from .base import SegmentationResult, Segmenter
from .dbscan import DBSCANSegmenter
from .region_growing import RegionGrowingSegmenter


def _extract_points(cloud: PointCloud) -> NDArray[np.float64]:
    """Extract X/Y/Z coordinates from a point cloud."""
    xs = [chunk[PointAttribute.X] for chunk in cloud]
    ys = [chunk[PointAttribute.Y] for chunk in cloud]
    zs = [chunk[PointAttribute.Z] for chunk in cloud]

    if not xs:
        return np.empty((0, 3), dtype=np.float64)

    return np.column_stack((np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)))


def _compute_relative_height(
    points: NDArray[np.float64],
    ground_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Height of each point above its geometrically NEAREST ground
    point (real (x, y, z) proximity, one query per point) -- not a
    single shared reference for the whole cloud. Mirrors the pattern
    already fixed in this same session for
    ``topocore.processing.features.geometric.RelativeHeightFeatureComputer``
    and already used correctly in
    ``topocore.processing.ground.manager.GroundManager.
    _nearest_ground_elevation()``.
    """
    ground_indices = np.flatnonzero(ground_mask)

    if ground_indices.size == 0:
        raise SegmentationError("No ground points found for relative height computation.")

    ground_points = points[ground_indices]
    manager = NeighborhoodManager.from_array(ground_points)

    # PR21.8 (transversal audit): same per-point query_point() loop
    # pattern already found and fixed in
    # features.geometric.RelativeHeightFeatureComputer,
    # classification.ml, and ground.manager -- found here via a
    # whole-processing/-tree search for this pattern. Replaced with
    # query_points_many(), numerically identical results (verified
    # directly before this change).
    indices, _ = manager.query_points_many(points, k=1)
    relative_height = points[:, 2] - cast(NDArray[np.float64], ground_points[indices[:, 0], 2])

    return relative_height


def _filter_cloud_by_relative_height(
    cloud: PointCloud,
    min_height: float,
    max_height: float,
    ground_method: str,
) -> tuple[PointCloud, NDArray[np.int64]]:
    """
    Filter a cloud by height ABOVE GROUND (not absolute Z) and
    return:
    - filtered cloud
    - indices in the original cloud corresponding to filtered points

    Found and fixed in PR19: an earlier version of this function
    (then named ``_filter_cloud_by_height``) filtered by raw,
    absolute Z -- despite every caller's own docstring describing
    ``min_height``/``max_height`` as "above ground", and
    ``TreeSegmenter``'s own class docstring explicitly listing
    "1. Ground classification to separate trees from ground" as the
    first step of its algorithm, a step that was never actually
    implemented. Confirmed directly: a realistic point cloud at
    ~1500m absolute elevation (ordinary real-world survey data, not
    ground-normalized) with a genuine 1-10m tall tree cluster raised
    ``SegmentationError: No points found above minimum height`` --
    the default ``min_height=0.5, max_height=50.0`` range never
    matches raw elevations in the thousands. Fixed by classifying
    ground first (``GroundManager``, already audited and cache-fixed
    elsewhere in this session) and filtering on each point's height
    above its own nearest ground point, not a single absolute range.
    """
    points = _extract_points(cloud)

    ground_mask = GroundManager(method=ground_method).classify(cloud)
    relative_height = _compute_relative_height(points, ground_mask)

    mask = (relative_height >= min_height) & (relative_height <= max_height)

    flattened = flatten_attributes(cloud)
    filtered_cloud = build_cloud_from_mask(flattened, mask)
    original_indices = np.flatnonzero(mask).astype(np.int64)

    return filtered_cloud, original_indices


def _build_labels_from_filtered_segments(
    original_point_count: int,
    filtered_to_original_indices: NDArray[np.int64],
    segments: list[NDArray[np.int64]],
) -> NDArray[np.int64]:
    """Build final labels in original-cloud index space."""
    labels = np.full(original_point_count, -1, dtype=np.int64)

    for segment_id, filtered_indices in enumerate(segments):
        original_indices = filtered_to_original_indices[filtered_indices]
        labels[original_indices] = segment_id

    return labels


class TreeSegmenter(Segmenter):
    """
    Specialized segmenter for individual trees.

    Uses a combination of:
    1. Ground classification to separate trees from ground
    2. Vertical connectivity to group tree points
    3. Height-based filtering to remove non-tree points

    Parameters
    ----------
    min_height
        Minimum height above ground for a point to be considered tree.
    max_height
        Maximum height for a tree point.
    eps
        DBSCAN epsilon for clustering.
    min_samples
        DBSCAN min_samples for clustering.
    min_points_per_tree
        Minimum points in a tree cluster.
    """

    __slots__ = (
        "_eps",
        "_ground_method",
        "_max_height",
        "_min_height",
        "_min_points_per_tree",
        "_min_samples",
    )

    def __init__(
        self,
        min_height: float = 0.5,
        max_height: float = 50.0,
        eps: float = 0.5,
        min_samples: int = 5,
        min_points_per_tree: int = 10,
        ground_method: str = "grid",
    ) -> None:
        if min_height < 0:
            raise SegmentationError(f"min_height must be >= 0, got {min_height}.")
        if max_height <= min_height:
            raise SegmentationError(f"max_height ({max_height}) must be > min_height ({min_height}).")
        if eps <= 0:
            raise SegmentationError(f"eps must be positive, got {eps}.")
        if min_samples < 1:
            raise SegmentationError(f"min_samples must be >= 1, got {min_samples}.")
        if min_points_per_tree < 1:
            raise SegmentationError(f"min_points_per_tree must be >= 1, got {min_points_per_tree}.")

        self._min_height = min_height
        self._max_height = max_height
        self._eps = eps
        self._min_samples = min_samples
        self._min_points_per_tree = min_points_per_tree
        self._ground_method = ground_method

    @override
    def segment(self, cloud: PointCloud) -> SegmentationResult:
        """Segment individual trees from the point cloud."""
        if cloud.is_empty:
            raise SegmentationError("Cannot segment an empty point cloud.")

        filtered_cloud, filtered_to_original = _filter_cloud_by_relative_height(
            cloud,
            self._min_height,
            self._max_height,
            self._ground_method,
        )

        if filtered_cloud.is_empty:
            raise SegmentationError("No points found above minimum height.")

        dbscan = DBSCANSegmenter(
            eps=self._eps,
            min_samples=self._min_samples,
            include_noise=True,
        )
        result = dbscan.segment(filtered_cloud)

        tree_segments = self._filter_tree_clusters(result)

        labels = _build_labels_from_filtered_segments(
            original_point_count=cloud.point_count,
            filtered_to_original_indices=filtered_to_original,
            segments=tree_segments,
        )

        return SegmentationResult(
            labels=labels,
            num_segments=len(tree_segments),
            segment_sizes=np.array([len(s) for s in tree_segments], dtype=np.int64),
            cloud=cloud,
        )

    def _filter_tree_clusters(
        self,
        result: SegmentationResult,
    ) -> list[NDArray[np.int64]]:
        """
        Filter clusters to keep only tree-like structures.

        Tree criteria:
        1. Vertical extent > horizontal extent
        2. Roughly cylindrical / vertical structure
        3. Minimum number of points
        """
        tree_segments: list[NDArray[np.int64]] = []

        for segment_id in range(result.num_segments):
            segment = result.get_segment(segment_id)

            if segment.point_count < self._min_points_per_tree:
                continue

            k = min(10, segment.point_count - 1)
            if k < 1:
                continue

            pca = PCAFeatures(k=k)
            features = pca.compute_all(segment)

            verticality = features.get("verticality")

            if verticality is None:
                continue

            if float(np.mean(verticality)) > 0.5:
                indices = np.flatnonzero(result.labels == segment_id).astype(np.int64)
                tree_segments.append(indices)

        return tree_segments

    @override
    @property
    def name(self) -> str:
        return f"tree_segmenter(height={self._min_height}-{self._max_height})"


class BuildingSegmenter(Segmenter):
    """
    Specialized segmenter for buildings.

    Uses region growing on smooth, planar surfaces with high
    geometric regularity.

    Parameters
    ----------
    min_height
        Minimum height above ground for a building point.
    max_height
        Maximum height for a building point.
    k
        Number of neighbors for normal estimation.
    curvature_threshold
        Maximum curvature for planar surfaces.
    normal_angle_threshold
        Maximum angle between normals for same surface.
    min_points_per_building
        Minimum points in a building.
    """

    __slots__ = (
        "_curvature_threshold",
        "_ground_method",
        "_k",
        "_max_height",
        "_min_height",
        "_min_points_per_building",
        "_normal_angle_threshold",
    )

    def __init__(
        self,
        min_height: float = 1.0,
        max_height: float = 100.0,
        k: int = 10,
        curvature_threshold: float = 0.02,
        normal_angle_threshold: float = 10.0,
        min_points_per_building: int = 100,
        ground_method: str = "grid",
    ) -> None:
        if min_height < 0:
            raise SegmentationError(f"min_height must be >= 0, got {min_height}.")
        if max_height <= min_height:
            raise SegmentationError(f"max_height ({max_height}) must be > min_height ({min_height}).")
        if k < 1:
            raise SegmentationError(f"k must be >= 1, got {k}.")
        if curvature_threshold < 0:
            raise SegmentationError(f"curvature_threshold must be >= 0, got {curvature_threshold}.")
        if normal_angle_threshold <= 0:
            raise SegmentationError(f"normal_angle_threshold must be > 0, got {normal_angle_threshold}.")
        if min_points_per_building < 1:
            raise SegmentationError(f"min_points_per_building must be >= 1, got {min_points_per_building}.")

        self._min_height = min_height
        self._max_height = max_height
        self._k = k
        self._curvature_threshold = curvature_threshold
        self._normal_angle_threshold = normal_angle_threshold
        self._min_points_per_building = min_points_per_building
        self._ground_method = ground_method

    @override
    def segment(self, cloud: PointCloud) -> SegmentationResult:
        """Segment buildings from the point cloud."""
        if cloud.is_empty:
            raise SegmentationError("Cannot segment an empty point cloud.")

        filtered_cloud, filtered_to_original = _filter_cloud_by_relative_height(
            cloud,
            self._min_height,
            self._max_height,
            self._ground_method,
        )

        if filtered_cloud.is_empty:
            raise SegmentationError("No points found in building height range.")

        rg = RegionGrowingSegmenter(
            k=self._k,
            curvature_threshold=self._curvature_threshold,
            normal_angle_threshold=self._normal_angle_threshold,
            min_region_size=self._min_points_per_building,
        )

        result = rg.segment(filtered_cloud)
        building_segments = self._filter_building_clusters(result)

        labels = _build_labels_from_filtered_segments(
            original_point_count=cloud.point_count,
            filtered_to_original_indices=filtered_to_original,
            segments=building_segments,
        )

        return SegmentationResult(
            labels=labels,
            num_segments=len(building_segments),
            segment_sizes=np.array([len(s) for s in building_segments], dtype=np.int64),
            cloud=cloud,
        )

    def _filter_building_clusters(
        self,
        result: SegmentationResult,
    ) -> list[NDArray[np.int64]]:
        """Filter clusters to keep only building-like structures."""
        building_segments: list[NDArray[np.int64]] = []

        for segment_id in range(result.num_segments):
            segment = result.get_segment(segment_id)

            if segment.point_count < self._min_points_per_building:
                continue

            k = min(10, segment.point_count - 1)
            if k < 1:
                continue

            pca = PCAFeatures(k=k)
            features = pca.compute_all(segment)

            planarity = features.get("planarity", np.array([0.0], dtype=np.float64))

            if float(np.mean(planarity)) > 0.5:
                indices = np.flatnonzero(result.labels == segment_id).astype(np.int64)
                building_segments.append(indices)

        return building_segments

    @override
    @property
    def name(self) -> str:
        return f"building_segmenter(height={self._min_height}-{self._max_height})"


__all__ = [
    "BuildingSegmenter",
    "TreeSegmenter",
]
