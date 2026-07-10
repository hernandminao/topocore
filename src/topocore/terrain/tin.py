"""
topocore.terrain.tin
====================

Triangulated Irregular Network (TIN).

This module defines the public terrain model used throughout TopoCore.
A TIN represents a terrain surface as an immutable collection of
vertices connected through a Delaunay triangulation.

The internal representation stores only vertex arrays and triangle
indices for maximum memory efficiency. High-level geometry objects
(Triangle, Edge) are generated lazily on demand.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.terrain.algorithms import DelaunayResult, DelaunayTriangulator
from topocore.terrain.models import Edge, Triangle


@dataclass(slots=True)
class _TINCache:
    """
    Internal cache.

    Expensive computations are cached the first time they are requested.
    """

    edges: tuple[Edge, ...] | None = None

    bounds: (
        tuple[
            float,
            float,
            float,
            float,
        ]
        | None
    ) = None


@dataclass(slots=True)
class TIN:
    """
    Immutable Triangulated Irregular Network.
    """

    _result: DelaunayResult

    _cache: _TINCache = field(
        default_factory=_TINCache,
        init=False,
        repr=False,
    )

    @classmethod
    def from_points(
        cls,
        points: list[Point3D] | tuple[Point3D, ...],
    ) -> TIN:
        """
        Build a TIN from terrain points.
        """
        return cls(
            DelaunayTriangulator.triangulate(points),
        )

    @property
    def vertices(
        self,
    ) -> tuple[Point3D, ...]:
        """
        Terrain vertices.
        """
        return self._result.vertices

    @property
    def simplices(
        self,
    ) -> NDArray[np.int32]:
        """
        Triangle connectivity.

        Returns
        -------
        ndarray
            Defensive copy.
        """
        return self._result.simplices.copy()

    @property
    def neighbors(
        self,
    ) -> NDArray[np.int32]:
        """
        Neighbor connectivity.

        Returns
        -------
        ndarray
            Defensive copy.
        """
        return self._result.neighbors.copy()

    @property
    def vertex_count(
        self,
    ) -> int:
        return self._result.vertex_count

    @property
    def triangle_count(
        self,
    ) -> int:
        return self._result.triangle_count

    @property
    def edge_count(
        self,
    ) -> int:
        return len(self.edges)

    @property
    def bounds(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        XY bounding box.
        """
        if self._cache.bounds is None:
            self._cache.bounds = DelaunayTriangulator.compute_bbox(
                self.vertices,
            )

        return self._cache.bounds

    def triangle(
        self,
        index: int,
    ) -> Triangle:
        """
        Return a triangle as a geometry object.

        Parameters
        ----------
        index
            Triangle index.

        Returns
        -------
        Triangle

        Raises
        ------
        IndexError
            If index is out of range.
        """
        simplex = self._result.simplices[index]

        vertices = self.vertices

        return Triangle(
            p1=vertices[int(simplex[0])],
            p2=vertices[int(simplex[1])],
            p3=vertices[int(simplex[2])],
        )

    def triangle_vertices(
        self,
        index: int,
    ) -> tuple[
        Point3D,
        Point3D,
        Point3D,
    ]:
        """
        Return the vertices of a triangle.

        Parameters
        ----------
        index
            Triangle index.

        Returns
        -------
        tuple[Point3D, Point3D, Point3D]
        """
        triangle = self.triangle(index)

        return (
            triangle.p1,
            triangle.p2,
            triangle.p3,
        )

    def triangles(
        self,
    ) -> Generator[Triangle]:
        """
        Iterate lazily over all triangles.

        Yields
        ------
        Triangle
        """
        for index in range(self.triangle_count):
            yield self.triangle(index)

    @property
    def edges(
        self,
    ) -> tuple[Edge, ...]:
        """
        Return all unique edges.

        The edge collection is generated lazily and cached.
        """
        if self._cache.edges is None:
            self._cache.edges = self._build_edges()

        return self._cache.edges

    def edge_vertices(
        self,
        edge: Edge,
    ) -> tuple[
        Point3D,
        Point3D,
    ]:
        """
        Return the vertices of an edge.
        """
        return (
            edge.start,
            edge.end,
        )

    def neighbors_of(
        self,
        triangle_index: int,
    ) -> tuple[
        int,
        int,
        int,
    ]:
        """
        Return neighboring triangle indices.
        """
        return DelaunayTriangulator.neighbor_indices(
            self._result,
            triangle_index,
        )

    def find_triangle(
        self,
        x: float,
        y: float,
    ) -> int:
        """
        Find the triangle containing a coordinate.
        """
        return DelaunayTriangulator.find_triangle(
            self._result,
            x,
            y,
        )

    def locate(
        self,
        point: Point3D,
    ) -> int:
        """
        Locate a point inside the TIN.
        """
        return self.find_triangle(
            point.x,
            point.y,
        )

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Return whether the coordinate lies inside the TIN.
        """
        return (
            self.find_triangle(
                x,
                y,
            )
            >= 0
        )

    def _build_edges(self) -> tuple[Edge, ...]:
        """
        Build the unique edge collection.

        Returns
        -------
        tuple[Edge, ...]
        """
        edge_indices: set[tuple[int, int]] = set()

        for simplex in self._result.simplices:
            i0 = int(simplex[0])
            i1 = int(simplex[1])
            i2 = int(simplex[2])

            edge_indices.add(tuple(sorted((i0, i1))))
            edge_indices.add(tuple(sorted((i1, i2))))
            edge_indices.add(tuple(sorted((i2, i0))))

        vertices = self.vertices

        return tuple(
            Edge(
                start=vertices[i],
                end=vertices[j],
            )
            for i, j in sorted(edge_indices)
        )

    def __iter__(
        self,
    ) -> Generator[Triangle]:
        """
        Iterate over all terrain triangles.

        Yields
        ------
        Triangle
        """
        yield from self.triangles()

    def __getitem__(
        self,
        index: int,
    ) -> Triangle:
        """
        Return a triangle by index.

        Parameters
        ----------
        index
            Triangle index.

        Returns
        -------
        Triangle
        """
        return self.triangle(index)

    def __len__(
        self,
    ) -> int:
        """
        Return the number of triangles.

        Returns
        -------
        int
        """
        return self.triangle_count

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation of the TIN.
        """
        return f"TIN(vertices={self.vertex_count}, triangles={self.triangle_count}, edges={self.edge_count})"

    def vertex_array(
        self,
    ) -> NDArray[np.float64]:
        """
        Return all vertices as an array.

        Returns
        -------
        ndarray
            Array of shape (n, 3).
        """
        return np.asarray(
            [
                (
                    point.x,
                    point.y,
                    point.z,
                )
                for point in self.vertices
            ],
            dtype=np.float64,
        )

    def xy_array(
        self,
    ) -> NDArray[np.float64]:
        """
        Return XY coordinates.

        Returns
        -------
        ndarray
            Array of shape (n, 2).
        """
        return np.asarray(
            [
                (
                    point.x,
                    point.y,
                )
                for point in self.vertices
            ],
            dtype=np.float64,
        )

    def elevation_array(
        self,
    ) -> NDArray[np.float64]:
        """
        Return all elevations.

        Returns
        -------
        ndarray
            Array of shape (n,).
        """
        return np.asarray(
            [point.z for point in self.vertices],
            dtype=np.float64,
        )


__all__ = [
    "TIN",
]
