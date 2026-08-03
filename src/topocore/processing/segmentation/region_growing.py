"""
topocore.processing.segmentation.region_growing
===============================================

Region Growing Segmentation.

This module implements the region growing algorithm for point cloud
segmentation. Points are grouped into regions based on smoothness
criteria (normal consistency, curvature threshold, etc.).

Region growing is particularly effective for segmenting:
- Planar surfaces (buildings, walls)
- Smooth surfaces (terrain)
- Objects with consistent orientation

References
----------
- Besl, P.J. and Jain, R.C. (1988). Segmentation through
  variable-order surface fitting. IEEE TPAMI, 10(2), 167-192.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, override

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.normals import NormalManager

from .base import RegionGrowingSegmenter as BaseRegionGrowingSegmenter
from .base import SegmentationResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class RegionGrowingSegmenter(BaseRegionGrowingSegmenter):
    """
    Region Growing segmentation for point clouds.

    Parameters
    ----------
    k
        Number of neighbors for normal estimation.
    curvature_threshold
        Maximum curvature for a point to be considered smooth.
    normal_angle_threshold
        Maximum angle, in degrees, between normals for two points
        to belong to the same region.
    min_region_size
        Minimum number of points for a valid region.
    max_region_size
        Maximum number of points for a valid region.
    use_adaptive_k
        If True, adapt k based on local density.

    Examples
    --------
    >>> segmenter = RegionGrowingSegmenter(
    ...     curvature_threshold=0.05,
    ...     normal_angle_threshold=15.0,
    ... )
    >>> result = segmenter.segment(cloud)
    >>> segments = result.get_segments()
    """

    __slots__ = (
        "_k",
        "_curvature_threshold",
        "_normal_angle_threshold",
        "_min_region_size",
        "_max_region_size",
        "_use_adaptive_k",
        "_normals",
        "_curvature",
        "_labels",
    )

    def __init__(
        self,
        k: int = 10,
        curvature_threshold: float = 0.05,
        normal_angle_threshold: float = 15.0,
        min_region_size: int = 10,
        max_region_size: int = 1_000_000,
        use_adaptive_k: bool = False,
    ) -> None:
        if k < 3:
            raise SegmentationError(f"k must be >= 3, got {k}.")

        if curvature_threshold < 0:
            raise SegmentationError(f"curvature_threshold must be non-negative, got {curvature_threshold}.")

        if not 0 <= normal_angle_threshold <= 90:
            raise SegmentationError(f"normal_angle_threshold must be in [0, 90], got {normal_angle_threshold}.")

        if min_region_size < 1:
            raise SegmentationError(f"min_region_size must be >= 1, got {min_region_size}.")

        if max_region_size < min_region_size:
            raise SegmentationError(
                f"max_region_size ({max_region_size}) must be >= min_region_size ({min_region_size})."
            )

        self._k = k
        self._curvature_threshold = curvature_threshold
        self._normal_angle_threshold = float(np.radians(normal_angle_threshold))
        self._min_region_size = min_region_size
        self._max_region_size = max_region_size
        self._use_adaptive_k = use_adaptive_k

        self._normals: FloatArray | None = None
        self._curvature: FloatArray | None = None
        self._labels: IntArray | None = None

    @override
    def segment(self, cloud: PointCloud) -> SegmentationResult:
        """Segment the point cloud using the region growing algorithm."""
        if cloud.is_empty:
            raise SegmentationError("Cannot segment an empty point cloud.")

        self._compute_normals(cloud)

        n_points = len(cloud)
        self._labels = np.full(n_points, -1, dtype=np.int64)
        region_id = 0

        manager = NeighborhoodManager.from_point_cloud(cloud)

        curvature = self._require_curvature()
        curvature_order = np.argsort(curvature)

        for idx_raw in curvature_order:
            idx = int(idx_raw)
            labels = self._require_labels()

            if labels[idx] != -1:
                continue

            labels[idx] = region_id
            self._grow_region(start_idx=idx, region_id=region_id, manager=manager)
            region_id += 1

        self._filter_small_regions()
        return self._build_result(cloud)

    def _compute_normals(self, cloud: PointCloud) -> None:
        """Compute normals and curvature for the point cloud."""

        if PointAttribute.NORMAL in cloud.attributes:
            # Use normals provided by the input cloud as-is.
            # Region growing assumes they are consistently oriented.
            # Opposite normal directions may reduce segmentation quality.

            normals_chunks: list[FloatArray] = []

            for chunk in cloud:
                normals = cast(FloatArray, chunk[PointAttribute.NORMAL])
                normals_chunks.append(normals)

            if not normals_chunks:
                raise SegmentationError(
                    "Point cloud contains NORMAL attribute metadata but no normal data chunks were found."
                )

            self._normals = np.concatenate(normals_chunks, axis=0)

            normal_manager = NormalManager(method="pca", k=self._k)
            _, self._curvature = normal_manager.estimate_both(cloud)
            return

        normal_manager = NormalManager(
            method="pca",
            k=self._k,
            orient_upward=True,
        )
        self._normals, self._curvature = normal_manager.estimate_both(cloud)

    def _grow_region(
        self,
        start_idx: int,
        region_id: int,
        manager: NeighborhoodManager,
    ) -> None:
        """Grow a region from a seed point."""
        seeds: list[int] = [start_idx]
        region_size = 1

        labels = self._require_labels()
        curvature = self._require_curvature()

        while seeds and region_size < self._max_region_size:
            current = seeds.pop()

            neighbors = manager.radius(current, radius=self._k * 0.1)

            for neighbor_raw in neighbors:
                neighbor = int(neighbor_raw)

                if not self._can_add_neighbor(
                    current,
                    neighbor,
                    labels,
                    curvature,
                ):
                    continue

                labels[neighbor] = region_id
                seeds.append(neighbor)
                region_size += 1

                if region_size >= self._max_region_size:
                    break

    def _can_add_neighbor(
        self,
        current: int,
        neighbor: int,
        labels: IntArray,
        curvature: FloatArray,
    ) -> bool:
        """Return whether a neighbor can be added to the current region."""
        if labels[neighbor] != -1:
            return False

        if curvature[neighbor] > self._curvature_threshold:
            return False

        return self._normals_consistent(current, neighbor)

    def _normals_consistent(self, idx1: int, idx2: int) -> bool:
        """
        Check whether two points have consistent normals.

        Returns
        -------
        bool
            True if the angle between normals is within the configured
            threshold.
        """
        normals = self._require_normals()
        n1 = normals[idx1]
        n2 = normals[idx2]

        dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        angle = float(np.arccos(dot))
        return angle <= self._normal_angle_threshold

    def _filter_small_regions(self) -> None:
        """Mark regions smaller than the minimum size as noise."""
        labels = self._require_labels()
        region_sizes = self._compute_region_sizes()

        for region_id, size in enumerate(region_sizes):
            if size < self._min_region_size:
                labels[labels == region_id] = -1

    def _compute_region_sizes(self) -> IntArray:
        """Compute the size of each non-negative region label."""
        labels = self._require_labels()

        valid_labels = labels[labels >= 0]
        if valid_labels.size == 0:
            return np.zeros(0, dtype=np.int64)

        max_id = int(np.max(valid_labels))
        sizes = np.zeros(max_id + 1, dtype=np.int64)

        for label_raw in valid_labels:
            label = int(label_raw)
            sizes[label] += 1

        return sizes

    def _build_result(self, cloud: PointCloud) -> SegmentationResult:
        """Build a segmentation result from the computed labels."""
        labels = self._require_labels()
        valid_labels = labels[labels >= 0]

        if valid_labels.size == 0:
            num_segments = 0
            segment_sizes = np.zeros(0, dtype=np.int64)
        else:
            num_segments = int(np.max(valid_labels)) + 1
            segment_sizes = np.zeros(num_segments, dtype=np.int64)

            for label_raw in valid_labels:
                label = int(label_raw)
                segment_sizes[label] += 1

        return SegmentationResult(
            labels=labels,
            num_segments=num_segments,
            segment_sizes=segment_sizes,
            cloud=cloud,
        )

    @override
    def seed_points(self, cloud: PointCloud) -> IntArray:
        """Select seed points ordered by increasing curvature."""
        self._compute_normals(cloud)
        curvature = self._require_curvature()
        return np.argsort(curvature)

    @override
    def smoothness_criterion(
        self,
        query_idx: int,
        neighbor_idx: int,
        cloud: PointCloud,
    ) -> bool:
        """
        Check whether a neighbor should be added to a region.

        Returns True when:
        1. Neighbor curvature is below the threshold.
        2. Normals are consistent.
        """
        if self._normals is None or self._curvature is None:
            self._compute_normals(cloud)

        curvature = self._require_curvature()
        if curvature[neighbor_idx] > self._curvature_threshold:
            return False

        return self._normals_consistent(query_idx, neighbor_idx)

    @property
    @override
    def name(self) -> str:
        """Return a human-readable segmenter name."""
        return f"region_growing(k={self._k}, curvature={self._curvature_threshold})"

    def _require_normals(self) -> FloatArray:
        """Return computed normals or raise if unavailable."""
        if self._normals is None:
            raise SegmentationError("Normals have not been computed.")
        return self._normals

    def _require_curvature(self) -> FloatArray:
        """Return computed curvature or raise if unavailable."""
        if self._curvature is None:
            raise SegmentationError("Curvature has not been computed.")
        return self._curvature

    def _require_labels(self) -> IntArray:
        """Return segmentation labels or raise if unavailable."""
        if self._labels is None:
            raise SegmentationError("Segmentation labels have not been initialized.")
        return self._labels


__all__: Sequence[str] = [
    "RegionGrowingSegmenter",
]
