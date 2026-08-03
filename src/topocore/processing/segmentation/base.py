"""
topocore.processing.segmentation.base
=====================================

Abstract base classes for point cloud segmentation.

This module defines the interfaces for:
- Segmenter: General segmentation interface
- RegionGrowingSegmenter: Region growing interface
- ClusterSegmenter: Clustering-based segmenter interface
- SegmentExtractor: Extracting individual segments

Segmentation is the process of grouping points into meaningful
clusters based on spatial proximity, geometric properties, or
semantic labels.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import build_cloud_from_mask as _build_cloud_from_mask
from topocore.processing._shared import flatten_attributes as _flatten_attributes
from topocore.processing.types import IntArray1D


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """
    Result of a segmentation operation.

    Attributes
    ----------
    labels
        Integer labels for each point (-1 for noise/unassigned).
    num_segments
        Number of segments found.
    segment_sizes
        Number of points in each segment.
    cloud
        The segmented point cloud (with labels added as an attribute).

    Notes
    -----
    Invariants enforced in ``__post_init__``:
    - ``labels`` must be a 1D integer array of length ``len(cloud)``.
    - ``num_segments`` must be non-negative.
    - ``segment_sizes`` must be a 1D integer array of shape
      ``(num_segments,)`` with non-negative entries.
    - Labels must be either ``-1`` (noise) or in ``[0, num_segments - 1]``.
    - The counts derived from ``labels`` must exactly match
      ``segment_sizes``.
    - Segment IDs must be contiguous from ``0`` to ``num_segments - 1``.
    """

    labels: IntArray1D
    num_segments: int
    segment_sizes: NDArray[np.int64]
    cloud: PointCloud

    def __post_init__(self) -> None:
        """Validate internal consistency of the segmentation result."""
        labels = np.asarray(self.labels)
        segment_sizes = np.asarray(self.segment_sizes)

        self._validate_num_segments()
        self._validate_labels(labels)
        self._validate_segment_sizes(segment_sizes)
        self._validate_label_domain(labels)
        self._validate_segment_sizes_consistency(labels, segment_sizes)
        self._validate_segment_id_contiguity(labels)

    def _validate_num_segments(self) -> None:
        """Validate number of segments."""
        if not isinstance(self.num_segments, int):
            raise TypeError(f"num_segments must be an int, got {type(self.num_segments).__name__}.")

        if self.num_segments < 0:
            raise ValueError(f"num_segments must be non-negative, got {self.num_segments}.")

    def _validate_labels(self, labels: np.ndarray) -> None:
        """Validate labels array."""
        if labels.ndim != 1:
            raise ValueError(f"labels must be a 1D array, got array with shape {labels.shape}.")

        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(f"labels must have an integer dtype, got {labels.dtype}.")

        n_points = len(self.cloud)

        if len(labels) != n_points:
            raise ValueError(f"labels length must match number of cloud points: {len(labels)} != {n_points}.")

    def _validate_segment_sizes(self, segment_sizes: np.ndarray) -> None:
        """Validate segment sizes array."""
        if segment_sizes.ndim != 1:
            raise ValueError(f"segment_sizes must be a 1D array, got shape {segment_sizes.shape}.")

        if not np.issubdtype(segment_sizes.dtype, np.integer):
            raise TypeError(f"segment_sizes must have an integer dtype, got {segment_sizes.dtype}.")

        if segment_sizes.shape != (self.num_segments,):
            raise ValueError(f"segment_sizes must have shape ({self.num_segments},), got {segment_sizes.shape}.")

        if np.any(segment_sizes < 0):
            raise ValueError("segment_sizes cannot contain negative values.")

    def _validate_label_domain(self, labels: np.ndarray) -> None:
        """Validate label values are inside allowed domain."""
        if labels.size == 0:
            return

        if self.num_segments == 0:
            invalid = labels[labels != -1]

            if invalid.size:
                raise ValueError(
                    "When num_segments == 0, all labels must be -1. "
                    f"Found invalid labels: {np.unique(invalid).tolist()}."
                )

            return

        invalid_mask = (labels < -1) | (labels >= self.num_segments)

        if np.any(invalid_mask):
            invalid = np.unique(labels[invalid_mask])

            raise ValueError(
                "labels contain values outside the valid range "
                f"{{-1, 0, ..., {self.num_segments - 1}}}: "
                f"{invalid.tolist()}."
            )

    def _validate_segment_sizes_consistency(
        self,
        labels: np.ndarray,
        segment_sizes: np.ndarray,
    ) -> None:
        """Ensure segment sizes match label counts."""
        non_noise = labels[labels >= 0]

        if self.num_segments == 0:
            actual_sizes = np.zeros(0, dtype=np.int64)
        else:
            actual_sizes = np.bincount(
                non_noise,
                minlength=self.num_segments,
            ).astype(np.int64, copy=False)

        if not np.array_equal(actual_sizes, segment_sizes):
            raise ValueError(
                "segment_sizes is inconsistent with labels. "
                f"Expected counts {actual_sizes.tolist()}, "
                f"got {segment_sizes.tolist()}."
            )

    def _validate_segment_id_contiguity(self, labels: np.ndarray) -> None:
        """Ensure segment IDs are contiguous."""
        if self.num_segments == 0:
            return

        non_noise = labels[labels >= 0]

        present_ids = np.unique(non_noise)
        expected_ids = np.arange(
            self.num_segments,
            dtype=present_ids.dtype,
        )

        if not np.array_equal(present_ids, expected_ids):
            raise ValueError(
                "Segment IDs must be contiguous and all referenced by labels. "
                f"Expected IDs {expected_ids.tolist()}, "
                f"found {present_ids.tolist()}."
            )

    def get_segment(self, segment_id: int) -> PointCloud:
        """
        Extract a single segment from the segmented cloud.

        Parameters
        ----------
        segment_id
            The segment ID to extract.

        Returns
        -------
        PointCloud
            A point cloud containing only the points of the segment.
        """
        if not isinstance(segment_id, int):
            raise TypeError(f"segment_id must be an int, got {type(segment_id).__name__}.")
        if not 0 <= segment_id < self.num_segments:
            raise ValueError(f"segment_id must be in [0, {self.num_segments - 1}], got {segment_id}.")

        mask = self.labels == segment_id
        flattened = _flatten_attributes(self.cloud)
        return _build_cloud_from_mask(flattened, mask)

    def get_segments(self, segment_ids: list[int] | None = None) -> list[PointCloud]:
        """
        Extract multiple segments.

        Parameters
        ----------
        segment_ids
            List of segment IDs to extract. If None, extracts all.

        Returns
        -------
        list[PointCloud]
            List of point clouds, one per segment.
        """
        ids = range(self.num_segments) if segment_ids is None else segment_ids
        return [self.get_segment(segment_id) for segment_id in ids]

    @property
    def has_noise(self) -> bool:
        """Return whether there are noise points (label = -1)."""
        return bool(np.any(self.labels == -1))


class Segmenter(ABC):
    """
    Abstract interface for point cloud segmenters.

    Segmenters group points into clusters based on spatial proximity,
    geometric properties, or semantic labels.

    Examples
    --------
    >>> segmenter = DBSCANSegmenter(eps=0.5, min_samples=5)
    >>> result = segmenter.segment(cloud)
    >>> segments = result.get_segments()
    """

    @abstractmethod
    def segment(self, cloud: PointCloud) -> SegmentationResult:
        """
        Segment the point cloud.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        SegmentationResult
            Result containing labels and segment information.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the segmenter name."""
        ...

    def __call__(self, cloud: PointCloud) -> SegmentationResult:
        """Callable interface."""
        return self.segment(cloud)


class ClusterSegmenter(Segmenter):
    """
    Abstract base class for clustering-based segmenters.

    Clustering segmenters group points based on spatial proximity
    (Euclidean distance) or density (DBSCAN, etc.).
    """

    @abstractmethod
    def cluster(self, points: NDArray[np.float64]) -> tuple[IntArray1D, int]:
        """
        Cluster points and return labels.

        Parameters
        ----------
        points
            Point coordinates (N, 3).

        Returns
        -------
        tuple
            (labels, num_clusters) where labels is an array of shape (N,)
            and num_clusters is the number of clusters found.
        """
        ...


class RegionGrowingSegmenter(Segmenter):
    """
    Abstract base class for region growing segmenters.

    Region growing segmenters start from seed points and iteratively
    add neighboring points that satisfy a smoothness criterion.
    """

    @abstractmethod
    def seed_points(self, cloud: PointCloud) -> IntArray1D:
        """
        Select seed points for region growing.

        Returns
        -------
        IntArray1D
            Indices of seed points.
        """
        ...

    @abstractmethod
    def smoothness_criterion(
        self,
        query_idx: int,
        neighbor_idx: int,
        cloud: PointCloud,
    ) -> bool:
        """
        Check if a neighbor should be added to the region.

        Parameters
        ----------
        query_idx
            Index of the current region point.
        neighbor_idx
            Index of the candidate neighbor.
        cloud
            The point cloud.

        Returns
        -------
        bool
            True if the neighbor should be added.
        """
        ...


__all__ = [
    "SegmentationResult",
    "Segmenter",
    "ClusterSegmenter",
    "RegionGrowingSegmenter",
]
