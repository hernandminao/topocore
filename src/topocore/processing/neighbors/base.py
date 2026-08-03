"""
topocore.processing.neighbors.base
==================================

Abstract base class for neighbor search operations.

This module defines the interface that all neighbor search implementations
must implement. The interface supports both KNN (k-nearest neighbors)
and radius-based searches, with batched and single-query variants.

The neighbor search is a fundamental component of the processing
subsystem, used by:
- Outlier removal filters (Statistical, Radius)
- Normal estimation
- Classification
- Segmentation
- Registration (ICP)

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.types import FloatArray2D, IntArray1D, IntArray2D


class NeighborSearch(ABC):
    """
    Abstract interface for spatial neighbor search.

    This interface provides methods for:
    - KNN (k-nearest neighbors) for single and multiple points
    - Radius search for single and multiple points
    - Batch operations for performance

    Implementations must be thread-safe and efficient for large
    point clouds.

    Examples
    --------
    >>> search = KDTreeNeighborSearch.from_point_cloud(cloud)
    >>> indices, distances = search.knn(0, k=10)
    >>> indices_batch, distances_batch = search.knn_many(k=10)
    """

    @abstractmethod
    def knn(
        self,
        index: int,
        k: int,
        *,
        include_self: bool = False,
    ) -> tuple[IntArray1D, NDArray[np.float64]]:
        """
        Find the k nearest neighbors of a single point.

        Parameters
        ----------
        index
            Index of the query point.
        k
            Number of neighbors to return.
        include_self
            If True, include the query point itself (distance 0).

        Returns
        -------
        tuple
            (neighbor_indices, distances) both of length k, sorted
            by increasing distance.

        Raises
        ------
        NeighborError
            If fewer than k unique neighbors are available.
        """
        ...

    @abstractmethod
    def knn_many(
        self,
        indices: IntArray1D | None = None,
        *,
        k: int,
        include_self: bool = False,
    ) -> tuple[IntArray2D, NDArray[np.float64]]:
        """
        Find the k nearest neighbors for multiple points.

        Parameters
        ----------
        indices
            Query point indices. If None, queries all points.
        k
            Number of neighbors per query.
        include_self
            If True, include each query point itself.

        Returns
        -------
        tuple
            (neighbor_indices, distances) both of shape (M, k),
            where M is the number of query points.

        Raises
        ------
        NeighborError
            If fewer than k unique neighbors are available.
        """
        ...

    @abstractmethod
    def radius(
        self,
        index: int,
        radius: float,
        *,
        include_self: bool = False,
    ) -> IntArray1D:
        """
        Find all neighbors of a single point within a radius.

        Parameters
        ----------
        index
            Index of the query point.
        radius
            Search radius.
        include_self
            If True, include the query point itself.

        Returns
        -------
        IntArray1D
            Indices of points within the radius (unsorted).
        """
        ...

    @abstractmethod
    def radius_many(
        self,
        indices: IntArray1D | None = None,
        *,
        radius: float,
        include_self: bool = False,
    ) -> list[IntArray1D]:
        """
        Find all neighbors for multiple points within a radius.

        Parameters
        ----------
        indices
            Query point indices. If None, queries all points.
        radius
            Search radius.
        include_self
            If True, include each query point itself.

        Returns
        -------
        list[IntArray1D]
            One array of neighbor indices per query point.
        """
        ...

    @abstractmethod
    def query_point(
        self,
        x: float,
        y: float,
        z: float,
        k: int,
    ) -> tuple[IntArray1D, NDArray[np.float64]]:
        """
        Find the k nearest neighbors to an arbitrary 3D point.

        Parameters
        ----------
        x, y, z
            Query point coordinates.
        k
            Number of neighbors to return.

        Returns
        -------
        tuple
            (neighbor_indices, distances) both of length k.
        """
        ...

    @abstractmethod
    def query_point_radius(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
    ) -> IntArray1D:
        """
        Find all neighbors to an arbitrary 3D point within a radius.

        Parameters
        ----------
        x, y, z
            Query point coordinates.
        radius
            Search radius.

        Returns
        -------
        IntArray1D
            Indices of points within the radius.
        """
        ...

    @property
    @abstractmethod
    def points(
        self,
    ) -> FloatArray2D:
        """
        The (N, 3) point array the index was built from.

        Returns
        -------
        FloatArray2D
            Array of shape (N, 3).
        """
        ...

    @property
    @abstractmethod
    def point_count(
        self,
    ) -> int:
        """
        Number of points in the index.

        Returns
        -------
        int
        """
        ...

    @classmethod
    @abstractmethod
    def from_point_cloud(
        cls,
        cloud: PointCloud,
    ) -> NeighborSearch:
        """
        Build a neighbor search index from a PointCloud.

        Parameters
        ----------
        cloud
            Source point cloud.

        Returns
        -------
        NeighborSearch
            An instance of the neighbor search implementation.
        """
        ...

    @classmethod
    @abstractmethod
    def from_array(
        cls,
        points: FloatArray2D,
    ) -> NeighborSearch:
        """
        Build a neighbor search index from an (N, 3) point array.

        Parameters
        ----------
        points
            Point coordinates, shape (N, 3).

        Returns
        -------
        NeighborSearch
            An instance of the neighbor search implementation.
        """
        ...


__all__ = [
    "NeighborSearch",
]
