"""
topocore.processing.segmentation.dbscan
=======================================

DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

This module implements the DBSCAN algorithm for point cloud segmentation.
DBSCAN groups points that are closely packed together (high density)
and marks points in low-density regions as noise.

References
----------
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).
  A density-based algorithm for discovering clusters in large
  spatial databases with noise. KDD.

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


class DBSCANSegmenter(ClusterSegmenter):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Parameters
    ----------
    eps
        Maximum distance between two points for them to be considered neighbors.
    min_samples
        Minimum number of points required to form a dense region.
    use_adaptive_eps
        Whether to adapt eps based on local density.
    include_noise
        Whether to keep noise points labeled as -1. If False, noise points
        are assigned to the nearest cluster when possible.
    cache_neighbors
        Whether to cache radius-neighborhood queries during clustering.
        This improves performance when neighborhood queries are expensive,
        at the cost of additional memory.
    """

    __slots__ = (
        "_eps",
        "_min_samples",
        "_use_adaptive_eps",
        "_include_noise",
        "_cache_neighbors",
    )

    _UNASSIGNED = -2
    _NOISE = -1

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        use_adaptive_eps: bool = False,
        include_noise: bool = True,
        cache_neighbors: bool = True,
    ) -> None:
        if eps <= 0:
            raise SegmentationError(f"eps must be positive, got {eps}.")

        if min_samples < 1:
            raise SegmentationError(f"min_samples must be >= 1, got {min_samples}.")

        self._eps = eps
        self._min_samples = min_samples
        self._use_adaptive_eps = use_adaptive_eps
        self._include_noise = include_noise
        self._cache_neighbors = cache_neighbors

    @override
    def segment(self, cloud: PointCloud) -> SegmentationResult:
        """Segment the point cloud using DBSCAN."""
        if cloud.is_empty:
            raise SegmentationError("Cannot segment an empty point cloud.")

        points = self._extract_points(cloud)
        labels, num_clusters = self.cluster(points)
        return self._build_result(cloud, labels, num_clusters)

    @override
    def cluster(self, points: NDArray[np.float64]) -> tuple[NDArray[np.int64], int]:
        """Cluster points using DBSCAN."""
        num_points = len(points)
        if num_points == 0:
            return np.empty(0, dtype=np.int64), 0

        manager = NeighborhoodManager.from_array(points)
        eps_values = self._compute_eps_values(points, manager)

        labels = np.full(num_points, self._UNASSIGNED, dtype=np.int64)
        visited = np.zeros(num_points, dtype=np.bool_)

        neighbor_cache: dict[int, list[int]] | None
        if self._cache_neighbors:
            neighbor_cache = {}
        else:
            neighbor_cache = None

        cluster_id = 0

        for point_index in range(num_points):
            if visited[point_index]:
                continue

            visited[point_index] = True
            neighbors = self._get_neighbors(
                manager=manager,
                point_index=point_index,
                eps_values=eps_values,
                neighbor_cache=neighbor_cache,
            )

            if not self._is_core_point(neighbors):
                labels[point_index] = self._NOISE
                continue

            self._expand_cluster(
                start_index=point_index,
                cluster_id=cluster_id,
                labels=labels,
                visited=visited,
                eps_values=eps_values,
                manager=manager,
                neighbor_cache=neighbor_cache,
            )
            cluster_id += 1

        if not self._include_noise:
            labels = self._assign_noise_to_nearest_cluster(labels, points, manager)

        return labels, cluster_id

    def _expand_cluster(
        self,
        start_index: int,
        cluster_id: int,
        labels: NDArray[np.int64],
        visited: NDArray[np.bool_],
        eps_values: NDArray[np.float64],
        manager: NeighborhoodManager,
        neighbor_cache: dict[int, list[int]] | None,
    ) -> None:
        """
        Expand a cluster from a core point.

        Uses a deque for efficient FIFO processing.
        """
        labels[start_index] = cluster_id

        frontier, frontier_set = self._initialize_frontier(
            start_index=start_index,
            manager=manager,
            eps_values=eps_values,
            neighbor_cache=neighbor_cache,
        )

        while frontier:
            current_index = frontier.popleft()
            frontier_set.discard(current_index)

            self._process_cluster_candidate(
                current_index=current_index,
                cluster_id=cluster_id,
                labels=labels,
                visited=visited,
                eps_values=eps_values,
                manager=manager,
                neighbor_cache=neighbor_cache,
                frontier=frontier,
                frontier_set=frontier_set,
            )

    def _initialize_frontier(
        self,
        start_index: int,
        manager: NeighborhoodManager,
        eps_values: NDArray[np.float64],
        neighbor_cache: dict[int, list[int]] | None,
    ) -> tuple[deque[int], set[int]]:
        """Create the initial expansion frontier."""
        neighbors = self._get_neighbors(
            manager=manager,
            point_index=start_index,
            eps_values=eps_values,
            neighbor_cache=neighbor_cache,
        )

        frontier = deque[int]()
        frontier_set: set[int] = set()

        for neighbor_index in neighbors:
            if neighbor_index != start_index:
                frontier.append(neighbor_index)
                frontier_set.add(neighbor_index)

        return frontier, frontier_set

    def _process_cluster_candidate(
        self,
        current_index: int,
        cluster_id: int,
        labels: NDArray[np.int64],
        visited: NDArray[np.bool_],
        eps_values: NDArray[np.float64],
        manager: NeighborhoodManager,
        neighbor_cache: dict[int, list[int]] | None,
        frontier: deque[int],
        frontier_set: set[int],
    ) -> None:
        """Process a candidate point during cluster expansion."""
        if not visited[current_index]:
            self._visit_candidate(
                current_index=current_index,
                labels=labels,
                visited=visited,
                eps_values=eps_values,
                manager=manager,
                neighbor_cache=neighbor_cache,
                frontier=frontier,
                frontier_set=frontier_set,
            )

        if labels[current_index] < 0:
            labels[current_index] = cluster_id

    def _visit_candidate(
        self,
        current_index: int,
        labels: NDArray[np.int64],
        visited: NDArray[np.bool_],
        eps_values: NDArray[np.float64],
        manager: NeighborhoodManager,
        neighbor_cache: dict[int, list[int]] | None,
        frontier: deque[int],
        frontier_set: set[int],
    ) -> None:
        """Visit a point and expand if it is a core point."""
        visited[current_index] = True

        neighbors = self._get_neighbors(
            manager=manager,
            point_index=current_index,
            eps_values=eps_values,
            neighbor_cache=neighbor_cache,
        )

        if self._is_core_point(neighbors):
            self._append_core_neighbors(
                neighbors=neighbors,
                labels=labels,
                frontier=frontier,
                frontier_set=frontier_set,
            )

    def _append_core_neighbors(
        self,
        neighbors: list[int],
        labels: NDArray[np.int64],
        frontier: deque[int],
        frontier_set: set[int],
    ) -> None:
        """Append new points from a core neighborhood."""
        for neighbor_index in neighbors:
            if self._should_expand_neighbor(
                neighbor_index,
                labels,
                frontier_set,
            ):
                frontier.append(neighbor_index)
                frontier_set.add(neighbor_index)

    def _should_expand_neighbor(
        self,
        neighbor_index: int,
        labels: NDArray[np.int64],
        frontier_set: set[int],
    ) -> bool:
        """Return whether a neighbor should enter the expansion queue."""
        if neighbor_index in frontier_set:
            return False

        return int(labels[neighbor_index]) < 0

    def _get_neighbors(
        self,
        manager: NeighborhoodManager,
        point_index: int,
        eps_values: NDArray[np.float64],
        neighbor_cache: dict[int, list[int]] | None,
    ) -> list[int]:
        """Return the neighbors of a point within its eps radius."""
        if neighbor_cache is not None:
            cached = neighbor_cache.get(point_index)
            if cached is not None:
                return cached

        neighbors = list(
            manager.radius(
                point_index,
                radius=eps_values[point_index],
                include_self=True,
            ),
        )

        if neighbor_cache is not None:
            neighbor_cache[point_index] = neighbors

        return neighbors

    def _is_core_point(self, neighbors: list[int]) -> bool:
        """Return whether the neighborhood satisfies the core-point condition."""
        return len(neighbors) >= self._min_samples

    def _compute_eps_values(
        self,
        points: NDArray[np.float64],
        manager: NeighborhoodManager,
    ) -> NDArray[np.float64]:
        """
        Compute eps values for each point.

        When adaptive eps is disabled, a constant eps is returned.
        When enabled, eps is scaled according to the local mean distance
        to the k nearest neighbors.
        """
        num_points = len(points)
        if not self._use_adaptive_eps:
            return np.full(num_points, self._eps, dtype=np.float64)

        if num_points <= 1:
            return np.full(num_points, self._eps, dtype=np.float64)

        k = min(self._min_samples * 2, num_points - 1)
        if k < 1:
            return np.full(num_points, self._eps, dtype=np.float64)

        _, distances = manager.knn_many(k=k, include_self=False)
        mean_distances = distances.mean(axis=1)

        global_mean_distance = mean_distances.mean()
        if global_mean_distance <= 0:
            return np.full(num_points, self._eps, dtype=np.float64)

        scale = mean_distances / global_mean_distance
        return self._eps * np.clip(scale, 0.5, 2.0)

    def _assign_noise_to_nearest_cluster(
        self,
        labels: NDArray[np.int64],
        points: NDArray[np.float64],
        manager: NeighborhoodManager,
    ) -> NDArray[np.int64]:
        """
        Assign noise points to the nearest non-noise cluster when possible.
        """
        noise_indices = np.flatnonzero(labels == self._NOISE)
        if len(noise_indices) == 0:
            return labels

        updated_labels = labels.copy()
        max_neighbors = min(100, len(points))

        for point_index_np in noise_indices:
            point_index = int(point_index_np)

            updated_labels[point_index] = self._find_nearest_cluster_label(
                point_index=point_index,
                labels=updated_labels,
                points=points,
                manager=manager,
                max_neighbors=max_neighbors,
            )

        return updated_labels

    def _find_nearest_cluster_label(
        self,
        point_index: int,
        labels: NDArray[np.int64],
        points: NDArray[np.float64],
        manager: NeighborhoodManager,
        max_neighbors: int,
    ) -> int:
        """
        Find the label of the nearest non-noise point.

        Returns NOISE when no clustered neighbor is found.
        """
        point = points[point_index]

        for k in range(1, max_neighbors):
            indices, _ = manager.query_point(point[0], point[1], point[2], k=k)
            for neighbor_index in indices:
                if labels[neighbor_index] >= 0:
                    return int(labels[neighbor_index])

        return self._NOISE

    def _extract_points(self, cloud: PointCloud) -> NDArray[np.float64]:
        """Extract X, Y and Z coordinates from the point cloud."""
        xs: list[NDArray[np.float64]] = []
        ys: list[NDArray[np.float64]] = []
        zs: list[NDArray[np.float64]] = []

        for chunk in cloud:
            xs.append(chunk[PointAttribute.X])
            ys.append(chunk[PointAttribute.Y])
            zs.append(chunk[PointAttribute.Z])

        return np.stack(
            (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)),
            axis=1,
        )

    def _build_result(
        self,
        cloud: PointCloud,
        labels: NDArray[np.int64],
        num_clusters: int,
    ) -> SegmentationResult:
        """Build a SegmentationResult from cluster labels."""
        segment_sizes = np.zeros(num_clusters, dtype=np.int64)

        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            segment_sizes[unique_labels] = counts

        return SegmentationResult(
            labels=labels,
            num_segments=num_clusters,
            segment_sizes=segment_sizes,
            cloud=cloud,
        )

    @override
    @property
    def name(self) -> str:
        return f"dbscan(eps={self._eps}, min_samples={self._min_samples})"


__all__ = ["DBSCANSegmenter"]
