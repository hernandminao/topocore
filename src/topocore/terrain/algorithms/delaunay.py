"""
topocore.terrain.algorithms.delaunay
====================================

Low-level Delaunay triangulation algorithm.

This module contains the computational backend used to generate
Triangulated Irregular Networks (TINs). The implementation is fully
independent from the public terrain models, allowing the triangulation
backend to evolve without affecting the public API.

Triangulation is performed in the XY plane while preserving Z
coordinates externally through point indexing.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from scipy.spatial import QhullError

from topocore.geometry.point3d import Point3D
from topocore.terrain.exceptions import TriangulationError

_MIN_POINTS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class DelaunayResult:
    """
    Immutable result produced by the triangulation algorithm.

    Parameters
    ----------
    vertices
        Original input vertices.

    simplices
        Triangle vertex indices.

    neighbors
        Neighbor triangle indices.
    """

    vertices: tuple[Point3D, ...]

    simplices: NDArray[np.int32]

    neighbors: NDArray[np.int32]

    @property
    def triangle_count(self) -> int:
        """Return the number of generated triangles."""
        return int(self.simplices.shape[0])

    @property
    def vertex_count(self) -> int:
        """Return the number of vertices."""
        return len(self.vertices)


class DelaunayTriangulator:
    """
    Computes a Delaunay triangulation.

    This class performs only the mathematical triangulation.
    It does not create TIN objects or any higher level terrain model.
    """

    __slots__ = ()

    @staticmethod
    def triangulate(
        points: list[Point3D] | tuple[Point3D, ...],
    ) -> DelaunayResult:
        """
        Compute the Delaunay triangulation.

        Parameters
        ----------
        points
            Terrain vertices.

        Returns
        -------
        DelaunayResult

        Raises
        ------
        TriangulationError
            If triangulation fails.
        """

        vertices = tuple(points)

        DelaunayTriangulator._validate(vertices)

        xy = DelaunayTriangulator._extract_xy(vertices)

        try:
            triangulation = Delaunay(
                xy,
                qhull_options="Qbb Qc Qz",
            )

        except QhullError as exc:
            raise TriangulationError(
                "Unable to compute Delaunay triangulation."
            ) from exc

        return DelaunayResult(
            vertices=vertices,
            simplices=np.asarray(
                triangulation.simplices,
                dtype=np.int32,
            ),
            neighbors=np.asarray(
                triangulation.neighbors,
                dtype=np.int32,
            ),
        )
    
    @staticmethod
    def _validate(
        vertices: tuple[Point3D, ...],
    ) -> None:
        """
        Validate the input point set.

        Parameters
        ----------
        vertices
            Input vertices.

        Raises
        ------
        TriangulationError
            If validation fails.
        """
        if len(vertices) < _MIN_POINTS:
            raise TriangulationError(
                "At least three points are required "
                "to compute a triangulation."
            )

        DelaunayTriangulator._validate_duplicate_points(
            vertices,
        )

        DelaunayTriangulator._validate_collinear_points(
            vertices,
        )

    @staticmethod
    def _extract_xy(
        vertices: tuple[Point3D, ...],
    ) -> NDArray[np.float64]:
        """
        Extract XY coordinates as a contiguous float64 array.

        Parameters
        ----------
        vertices
            Input vertices.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 2).
        """
        xy = np.empty(
            (len(vertices), 2),
            dtype=np.float64,
        )

        for index, point in enumerate(vertices):
            xy[index, 0] = point.x
            xy[index, 1] = point.y

        return xy

    @staticmethod
    def _validate_duplicate_points(
        vertices: tuple[Point3D, ...],
    ) -> None:
        """
        Detect duplicated XY coordinates.

        Parameters
        ----------
        vertices
            Input vertices.

        Raises
        ------
        TriangulationError
            If duplicated XY coordinates are found.
        """
        seen: set[tuple[float, float]] = set()

        for point in vertices:
            key = (point.x, point.y)

            if key in seen:
                raise TriangulationError(
                    "Duplicated XY coordinates were found "
                    "in the input point set."
                )

            seen.add(key)

    @staticmethod
    def _validate_collinear_points(
        vertices: tuple[Point3D, ...],
    ) -> None:
        """
        Detect fully collinear datasets.

        Delaunay triangulation requires a 2D point cloud.
        If all XY coordinates are collinear the triangulation
        cannot be computed.

        Parameters
        ----------
        vertices
            Input vertices.

        Raises
        ------
        TriangulationError
            If all points are collinear.
        """
        xy = DelaunayTriangulator._extract_xy(vertices)

        centered = xy - xy.mean(axis=0)

        rank = np.linalg.matrix_rank(centered)

        if rank < 2:
            raise TriangulationError(
                "All points are collinear."
            )

    @staticmethod
    def compute_bbox(
        vertices: tuple[Point3D, ...],
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Compute the XY bounding box.

        Parameters
        ----------
        vertices
            Input vertices.

        Returns
        -------
        tuple
            (min_x, min_y, max_x, max_y)
        """
        min_x = min(point.x for point in vertices)
        min_y = min(point.y for point in vertices)
        max_x = max(point.x for point in vertices)
        max_y = max(point.y for point in vertices)

        return (
            min_x,
            min_y,
            max_x,
            max_y,
        )
    
    @staticmethod
    def validate_result(
        result: DelaunayResult,
    ) -> None:
        """
        Validate the consistency of a triangulation result.

        This method is primarily intended for internal use and unit
        testing.

        Parameters
        ----------
        result
            Triangulation result.

        Raises
        ------
        TriangulationError
            If the result is inconsistent.
        """
        vertex_count = result.vertex_count

        if result.simplices.ndim != 2:
            raise TriangulationError(
                "Invalid simplices array."
            )

        if result.simplices.shape[1] != 3:
            raise TriangulationError(
                "Each simplex must contain exactly three vertices."
            )

        if result.neighbors.shape != result.simplices.shape:
            raise TriangulationError(
                "Neighbors array has an invalid shape."
            )

        if np.any(result.simplices < 0):
            raise TriangulationError(
                "Triangulation contains invalid indices."
            )

        if np.any(result.simplices >= vertex_count):
            raise TriangulationError(
                "Triangulation references nonexistent vertices."
            )

    @staticmethod
    def triangle_vertices(
        result: DelaunayResult,
        triangle_index: int,
    ) -> tuple[int, int, int]:
        """
        Return the vertex indices of a triangle.

        Parameters
        ----------
        result
            Triangulation result.

        triangle_index
            Triangle index.

        Returns
        -------
        tuple[int, int, int]

        Raises
        ------
        IndexError
            If the triangle index is invalid.
        """
        simplex = result.simplices[triangle_index]

        return (
            int(simplex[0]),
            int(simplex[1]),
            int(simplex[2]),
        )

    @staticmethod
    def neighbor_indices(
        result: DelaunayResult,
        triangle_index: int,
    ) -> tuple[int, int, int]:
        """
        Return neighboring triangle indices.

        Parameters
        ----------
        result
            Triangulation result.

        triangle_index
            Triangle index.

        Returns
        -------
        tuple[int, int, int]

        Notes
        -----
        A value of -1 indicates that the corresponding edge lies on the
        boundary of the convex hull.
        """
        neighbors = result.neighbors[triangle_index]

        return (
            int(neighbors[0]),
            int(neighbors[1]),
            int(neighbors[2]),
        )


__all__ = [
    "DelaunayResult",
    "DelaunayTriangulator",
]