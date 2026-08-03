"""
topocore.processing.segmentation.connected_components
=====================================================

Connected Components Segmentation.

This module implements connected components clustering for point clouds.
Points are connected if they are within a distance threshold of each
other. This is the simplest form of segmentation and works well for:

- Isolated objects
- Objects with clear spatial separation
- Quick initial segmentation

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections import deque
from typing import override

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.neighbors import NeighborhoodManager

from .base import ClusterSegmenter, SegmentationResult


class ConnectedComponentsSegmenter(ClusterSegmenter):
    """
    Connected Components Segmentation.

    Groups points that are connected through spatial proximity.

    Parameters
    ----------
    distance_threshold
        Maximum distance for two points to be considered connected.
    min_points
        Minimum number of points for a valid component.
    use_adaptive_threshold
        If True, adapt threshold based on local density.

    Examples
    --------
    >>> segmenter = ConnectedComponentsSegmenter(distance_threshold=0.5)
    >>> result = segmenter.segment(cloud)
    >>> segments = result.get_segments()
    """

    __slots__ = (
        "_distance_threshold",
        "_min_points",
        "_use_adaptive_threshold",
    )

    def __init__(
        self,
        distance_threshold: float = 0.5,
        min_points: int = 10,
        use_adaptive_threshold: bool = False,
    ) -> None:
        if distance_threshold <= 0:
            raise SegmentationError(f"distance_threshold must be positive, got {distance_threshold}.")

        if min_points < 1:
            raise SegmentationError(f"min_points must be >= 1, got {min_points}.")

        self._distance_threshold = distance_threshold
        self._min_points = min_points
        self._use_adaptive_threshold = use_adaptive_threshold

    @override
    def segment(
        self,
        cloud: PointCloud,
    ) -> SegmentationResult:
        """Segment the point cloud using connected components."""
        if cloud.is_empty:
            raise SegmentationError("Cannot segment an empty point cloud.")

        # Extract points
        points = self._extract_points(cloud)

        # Cluster
        labels, num_clusters = self.cluster(points)

        # Build result
        return self._build_result(cloud, labels, num_clusters)

    from collections import deque

    @override
    def cluster(
        self,
        points: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], int]:
        """Cluster points using connected components."""
        n = points.shape[0]
        labels = np.full(n, -1, dtype=np.int64)
        cluster_id = 0

        manager = NeighborhoodManager.from_array(points)
        threshold_values = self._compute_threshold_values(points, manager)

        for i in range(n):
            if labels[i] != -1:
                continue

            current_label = cluster_id
            labels[i] = current_label

            queue = deque([i])
            component_indices = [i]

            while queue:
                current = queue.popleft()

                threshold = threshold_values[current]
                neighbors = manager.radius(current, radius=threshold, include_self=True)

                for neighbor in neighbors:
                    if labels[neighbor] == -1:
                        labels[neighbor] = current_label
                        queue.append(neighbor)
                        component_indices.append(neighbor)

            if len(component_indices) < self._min_points:
                labels[component_indices] = -1
            else:
                cluster_id += 1

        return labels, cluster_id

    def _compute_threshold_values(
        self,
        points: NDArray[np.float64],
        manager: NeighborhoodManager,
    ) -> NDArray[np.float64]:
        """
        Compute adaptive threshold values based on local density.
        """
        n = points.shape[0]

        if not self._use_adaptive_threshold:
            return np.full(n, self._distance_threshold, dtype=np.float64)

        if n <= 1:
            return np.full(n, self._distance_threshold, dtype=np.float64)

        k = min(5, n - 1)
        if k == 0:
            return np.full(n, self._distance_threshold, dtype=np.float64)

        _, distances = manager.knn_many(k=k, include_self=False)
        mean_dist = distances.mean(axis=1)

        global_mean = mean_dist.mean()
        if global_mean <= 0:
            return np.full(n, self._distance_threshold, dtype=np.float64)

        scale = mean_dist / global_mean
        return self._distance_threshold * np.clip(scale, 0.5, 2.0)

    def _extract_points(
        self,
        cloud: PointCloud,
    ) -> NDArray[np.float64]:
        """Extract X/Y/Z coordinates from a point cloud."""
        xs, ys, zs = [], [], []
        for chunk in cloud:
            xs.append(chunk[PointAttribute.X])
            ys.append(chunk[PointAttribute.Y])
            zs.append(chunk[PointAttribute.Z])

        return np.column_stack(
            (
                np.concatenate(xs),
                np.concatenate(ys),
                np.concatenate(zs),
            )
        )

    def _build_result(
        self,
        cloud: PointCloud,
        labels: NDArray[np.int64],
        num_clusters: int,
    ) -> SegmentationResult:
        """Build a SegmentationResult from labels."""
        # Build segment sizes
        if num_clusters == 0:
            segment_sizes = np.zeros(0, dtype=np.int64)
        else:
            valid_labels = labels[labels >= 0]
            segment_sizes = np.bincount(valid_labels, minlength=num_clusters)

        if segment_sizes.dtype != np.int64:
            segment_sizes = segment_sizes.astype(np.int64, copy=False)

        return SegmentationResult(
            labels=labels,
            num_segments=num_clusters,
            segment_sizes=segment_sizes,
            cloud=cloud,
        )

    @override
    @property
    def name(self) -> str:
        return f"connected_components(threshold={self._distance_threshold})"


__all__ = [
    "ConnectedComponentsSegmenter",
]
