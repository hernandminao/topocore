"""
topocore.processing.neighbors.kdtree
====================================

KD-tree implementation of the NeighborSearch interface.

This module provides a KD-tree-based neighbor search using
`scipy.spatial.cKDTree`. The KD-tree is a space-partitioning data
structure that enables efficient O(log N) nearest neighbor queries
for low-dimensional data (3D point clouds).

The implementation uses `scipy.spatial.cKDTree` because it is:
- Battle-tested and highly optimized (C++ backend)
- Permissively licensed (BSD)
- Already a dependency of TopoCore (via scipy)
- Supports both KNN and radius queries efficiently

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import cast, final, override

import numpy as np
from numpy.typing import NDArray

# mypy: disable-error-code=import-untyped
from scipy.spatial import cKDTree

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import NeighborError
from topocore.processing.types import (
    FloatArray2D,
    IntArray1D,
    IntArray2D,
)

from .base import NeighborSearch


def _extract_xyz(
    cloud: PointCloud,
) -> FloatArray2D:
    """
    Extract X/Y/Z coordinates from a PointCloud.

    Returns
    -------
    FloatArray2D
        Array with shape (N, 3).
    """

    if cloud.is_empty:
        raise NeighborError("Cannot build a neighbor index from an empty point cloud.")

    required = (
        PointAttribute.X,
        PointAttribute.Y,
        PointAttribute.Z,
    )

    if not all(attribute in cloud.attributes for attribute in required):
        raise NeighborError("Point cloud must contain X/Y/Z coordinates.")

    xs: list[NDArray[np.float64]] = []
    ys: list[NDArray[np.float64]] = []
    zs: list[NDArray[np.float64]] = []

    for chunk in cloud:
        xs.append(
            cast(
                NDArray[np.float64],
                chunk[PointAttribute.X],
            )
        )

        ys.append(
            cast(
                NDArray[np.float64],
                chunk[PointAttribute.Y],
            )
        )

        zs.append(
            cast(
                NDArray[np.float64],
                chunk[PointAttribute.Z],
            )
        )

    x = np.concatenate(xs).astype(
        np.float64,
        copy=False,
    )

    y = np.concatenate(ys).astype(
        np.float64,
        copy=False,
    )

    z = np.concatenate(zs).astype(
        np.float64,
        copy=False,
    )

    return cast(
        FloatArray2D,
        np.stack(
            [
                x,
                y,
                z,
            ],
            axis=1,
        ),
    )


def _drop_self_column(
    neighbor_indices: IntArray2D,
    distances: NDArray[np.float64],
    query_indices: IntArray1D,
    k: int,
) -> tuple[IntArray2D, NDArray[np.float64]]:
    """
    Remove query points from their own neighbor results.
    """

    is_self = neighbor_indices == query_indices[:, None]

    if not is_self.any(axis=1).all():
        raise NeighborError("Could not find the query point among its own neighbors.")

    self_column = is_self.argmax(axis=1)

    columns = np.arange(
        neighbor_indices.shape[1],
    )[None, :]

    keep = columns != self_column[:, None]

    rows = neighbor_indices.shape[0]

    filtered_indices = neighbor_indices[keep].reshape(
        rows,
        -1,
    )

    filtered_distances = distances[keep].reshape(
        rows,
        -1,
    )

    if filtered_indices.shape[1] < k:
        raise NeighborError(f"Could not obtain {k} unique neighbors.")

    return (
        cast(
            IntArray2D,
            filtered_indices[:, :k],
        ),
        cast(
            NDArray[np.float64],
            filtered_distances[:, :k],
        ),
    )


@final
class KDTreeNeighborSearch(NeighborSearch):
    __slots__ = (
        "_points",
        "_tree",
        "_workers",
    )

    def __init__(
        self,
        points: FloatArray2D,
        *,
        workers: int = 1,
    ) -> None:

        if points.ndim != 2 or points.shape[1] != 3:
            raise NeighborError(f"points must have shape (N,3), got {points.shape}")

        if points.shape[0] == 0:
            raise NeighborError("Cannot create KD-tree with zero points.")

        self._validate_workers(workers)

        self._points = cast(
            FloatArray2D,
            points.astype(
                np.float64,
                copy=False,
            ),
        )

        self._tree = cKDTree(
            self._points,
        )

        # PR21.2: number of worker threads scipy.spatial.cKDTree.query()/
        # query_ball_point() use internally. Default (1) is scipy's own
        # default and preserves EXACTLY the pre-PR21 single-threaded
        # behavior/timing -- this is opt-in, never forced on a caller.
        # -1 uses all available CPU cores. Each query point's neighbor
        # search is fully independent of every other's, so parallelizing
        # this batch cannot change the numerical result -- only wall-
        # clock time -- confirmed by this module's own regression suite
        # (workers=1 and workers=-1 produce identical indices/distances
        # for the same input). Not benchmarked as a speedup in a
        # single-core environment (no additional cores to parallelize
        # across there), but zero-risk to enable given scipy's own
        # documented, battle-tested parallel-query support.
        self._workers = int(workers)

    @classmethod
    @override
    def from_point_cloud(
        cls,
        cloud: PointCloud,
        *,
        workers: int = 1,
    ) -> KDTreeNeighborSearch:
        return cls(_extract_xyz(cloud), workers=workers)

    @classmethod
    @override
    def from_array(
        cls,
        points: FloatArray2D,
        *,
        workers: int = 1,
    ) -> KDTreeNeighborSearch:
        return cls(points, workers=workers)

    @override
    def knn(
        self,
        index: int,
        k: int,
        *,
        include_self: bool = False,
    ) -> tuple[IntArray1D, NDArray[np.float64]]:

        self._validate_index(index)
        self._validate_k(k)

        query_k = k if include_self else min(k + 1, self.point_count)

        distances_raw, indices_raw = self._tree.query(
            self._points[index],
            k=query_k,
            workers=self._workers,
        )

        distances = cast(
            NDArray[np.float64],
            np.atleast_1d(distances_raw),
        )

        indices = cast(
            IntArray1D,
            np.atleast_1d(indices_raw).astype(
                np.int64,
                copy=False,
            ),
        )

        if not include_self:
            filtered_indices, filtered_distances = _drop_self_column(
                indices.reshape(1, -1),
                distances.reshape(1, -1),
                np.asarray(
                    [index],
                    dtype=np.int64,
                ),
                k,
            )

            return (
                filtered_indices[0],
                filtered_distances[0],
            )

        return (
            indices,
            distances,
        )

    @override
    def knn_many(
        self,
        indices: IntArray1D | None = None,
        *,
        k: int,
        include_self: bool = False,
    ) -> tuple[IntArray2D, NDArray[np.float64]]:

        self._validate_k(k)

        query_indices: IntArray1D = np.asarray(
            (
                np.arange(
                    self.point_count,
                    dtype=np.int64,
                )
                if indices is None
                else indices
            ),
            dtype=np.int64,
        )

        self._validate_indices(
            query_indices,
        )

        query_k = k if include_self else min(k + 1, self.point_count)

        distances_raw, neighbor_indices_raw = self._tree.query(
            self._points[query_indices],
            k=query_k,
            workers=self._workers,
        )

        distances = cast(
            NDArray[np.float64],
            np.atleast_1d(distances_raw),
        )

        neighbor_indices = cast(
            IntArray2D,
            np.atleast_1d(neighbor_indices_raw).astype(
                np.int64,
                copy=False,
            ),
        )

        # scipy returns (M,) instead of (M,1) when k=1
        if query_k == 1:
            distances = cast(
                NDArray[np.float64],
                distances.reshape(
                    -1,
                    1,
                ),
            )

            neighbor_indices = cast(
                IntArray2D,
                neighbor_indices.reshape(
                    -1,
                    1,
                ),
            )

        if not include_self:
            neighbor_indices, distances = _drop_self_column(
                neighbor_indices,
                distances,
                query_indices,
                k,
            )

        return (
            neighbor_indices,
            distances,
        )

    @override
    def radius(
        self,
        index: int,
        radius: float,
        *,
        include_self: bool = False,
    ) -> IntArray1D:

        self._validate_index(index)
        self._validate_radius(radius)

        neighbors = np.asarray(
            self._tree.query_ball_point(
                self._points[index],
                radius,
                workers=self._workers,
            ),
            dtype=np.int64,
        )

        if not include_self:
            neighbors = neighbors[neighbors != index]

        return cast(
            IntArray1D,
            neighbors,
        )

    @override
    def radius_many(
        self,
        indices: IntArray1D | None = None,
        *,
        radius: float,
        include_self: bool = False,
    ) -> list[IntArray1D]:

        self._validate_radius(radius)

        query_indices: IntArray1D = np.asarray(
            (
                np.arange(
                    self.point_count,
                    dtype=np.int64,
                )
                if indices is None
                else indices
            ),
            dtype=np.int64,
        )

        self._validate_indices(
            query_indices,
        )

        query_points = self._points[query_indices]

        raw_results = self._tree.query_ball_point(
            query_points,
            radius,
            workers=self._workers,
        )

        results: list[IntArray1D] = [
            cast(
                IntArray1D,
                np.asarray(
                    result,
                    dtype=np.int64,
                ),
            )
            for result in raw_results
        ]

        if not include_self:
            results = [
                result[result != query]
                for result, query in zip(
                    results,
                    query_indices,
                    strict=True,
                )
            ]

        return results

    @override
    def query_point(
        self,
        x: float,
        y: float,
        z: float,
        k: int,
    ) -> tuple[IntArray1D, NDArray[np.float64]]:

        self._validate_k(k)

        distances_raw, indices_raw = self._tree.query(
            [
                x,
                y,
                z,
            ],
            k=k,
            workers=self._workers,
        )

        distances = cast(
            NDArray[np.float64],
            np.atleast_1d(distances_raw),
        )

        indices = cast(
            IntArray1D,
            np.atleast_1d(indices_raw).astype(
                np.int64,
                copy=False,
            ),
        )

        return (
            indices,
            distances,
        )

    @override
    def query_point_radius(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
    ) -> IntArray1D:

        self._validate_radius(radius)

        neighbors = np.asarray(
            self._tree.query_ball_point(
                [
                    x,
                    y,
                    z,
                ],
                radius,
                workers=self._workers,
            ),
            dtype=np.int64,
        )

        return cast(
            IntArray1D,
            neighbors,
        )

    @property
    @override
    def points(
        self,
    ) -> FloatArray2D:

        return self._points.copy()

    @property
    @override
    def point_count(
        self,
    ) -> int:

        return int(self._points.shape[0])

    def _validate_index(
        self,
        index: int,
    ) -> None:

        if not isinstance(
            index,
            (int, np.integer),
        ):
            raise NeighborError(f"index must be integer, got {type(index).__name__}")

        if index < 0 or index >= self.point_count:
            raise NeighborError(f"index {index} outside valid range.")

    def _validate_indices(
        self,
        indices: IntArray1D,
    ) -> None:

        if (indices < 0).any() or (indices >= self.point_count).any():
            raise NeighborError("One or more indices are out of bounds.")

    @staticmethod
    def _validate_k(
        k: int,
    ) -> None:

        if not isinstance(
            k,
            (int, np.integer),
        ):
            raise NeighborError(f"k must be integer, got {type(k).__name__}")

        if k < 1:
            raise NeighborError(f"k must be >= 1, got {k}")

        if k > 1_000_000:
            raise NeighborError(f"k={k} exceeds maximum allowed.")

    @staticmethod
    def _validate_workers(
        workers: int,
    ) -> None:
        """
        Validate the `workers` parameter before it ever reaches
        scipy, matching this module's own convention (see
        `_validate_k`/`_validate_radius`) of converting invalid
        input into a `NeighborError` at the TopoCore boundary rather
        than letting a raw third-party exception leak through.
        scipy.spatial.cKDTree.query()'s own contract: -1 means "use
        all available CPUs", any other value must be a positive
        integer (a specific worker count); 0 and negative values
        other than -1 are invalid.
        """

        if not isinstance(
            workers,
            (int, np.integer),
        ):
            raise NeighborError(f"workers must be integer, got {type(workers).__name__}")

        if workers == 0 or workers < -1:
            raise NeighborError(f"workers must be -1 or >= 1, got {workers}")

    @staticmethod
    def _validate_radius(
        radius: float,
    ) -> None:

        if not np.isfinite(radius):
            raise NeighborError(f"radius must be finite, got {radius}")

        if radius <= 0:
            raise NeighborError(f"radius must be positive, got {radius}")


__all__ = [
    "KDTreeNeighborSearch",
]
