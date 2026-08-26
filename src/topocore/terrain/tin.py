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

import math
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.terrain.algorithms import DelaunayResult, DelaunayTriangulator
from topocore.terrain.constants import EPSILON
from topocore.terrain.exceptions import TriangulationError
from topocore.terrain.models import Edge, Triangle


def _orientation(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    """
    Twice the signed area of triangle (a, b, c).
    """
    return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)


def _point_in_triangle(
    x: float,
    y: float,
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
) -> bool:
    """
    True if (x, y) lies inside or on the boundary of triangle
    (p1, p2, p3).

    Standard barycentric sign test: the point is inside exactly when
    it is not strictly on both sides (positive and negative) of the
    three edges at once.
    """
    d1 = _orientation(x, y, p1.x, p1.y, p2.x, p2.y)
    d2 = _orientation(x, y, p2.x, p2.y, p3.x, p3.y)
    d3 = _orientation(x, y, p3.x, p3.y, p1.x, p1.y)

    has_negative = (d1 < 0.0) or (d2 < 0.0) or (d3 < 0.0)
    has_positive = (d1 > 0.0) or (d2 > 0.0) or (d3 > 0.0)

    return not (has_negative and has_positive)


def _build_neighbors(
    simplices: NDArray[np.int32],
) -> NDArray[np.int32]:
    """
    Compute triangle adjacency from explicit triangle connectivity.

    Reuses the same edge-sharing idea already used by
    ``terrain.algorithms.constrained_delaunay._build_mesh``: two
    triangles are neighbors exactly when they share an edge (a pair
    of vertex indices). No ``scipy.spatial.Delaunay`` call is
    involved -- this only reads the given connectivity.

    Matches the ``scipy.spatial.Delaunay.neighbors`` convention used
    everywhere else in ``TIN``: ``neighbors[i, j]`` is the neighbor
    of triangle ``i`` across the edge opposite vertex ``j`` (i.e.
    the edge formed by the *other* two vertices of the triangle).

    Parameters
    ----------
    simplices
        Triangle vertex indices, shape ``(n, 3)``.

    Returns
    -------
    ndarray
        Neighbor triangle indices, shape ``(n, 3)``. ``-1`` marks a
        boundary edge (no neighbor, or an edge shared by more than
        two triangles -- non-manifold input is treated as boundary
        rather than guessed at).
    """
    triangle_count = simplices.shape[0]

    edge_map: dict[frozenset[int], list[int]] = {}

    for triangle_index in range(triangle_count):
        v0, v1, v2 = (int(simplices[triangle_index, k]) for k in range(3))

        # Edge opposite vertex j is the edge formed by the other two.
        opposite_edges = (
            frozenset((v1, v2)),  # opposite v0
            frozenset((v0, v2)),  # opposite v1
            frozenset((v0, v1)),  # opposite v2
        )

        for edge in opposite_edges:
            edge_map.setdefault(edge, []).append(triangle_index)

    neighbors = np.full((triangle_count, 3), -1, dtype=np.int32)

    for triangle_index in range(triangle_count):
        v0, v1, v2 = (int(simplices[triangle_index, k]) for k in range(3))

        opposite_edges = (
            frozenset((v1, v2)),
            frozenset((v0, v2)),
            frozenset((v0, v1)),
        )

        for local_j, edge in enumerate(opposite_edges):
            sharing = edge_map[edge]

            if len(sharing) == 2:
                other = sharing[0] if sharing[1] == triangle_index else sharing[1]
                neighbors[triangle_index, local_j] = other

    return neighbors


def _validate_mesh(
    vertices: tuple[Point3D, ...],
    simplices: NDArray[np.int32],
) -> None:
    """
    Validate explicit vertex/triangle connectivity before building a
    ``TIN`` via ``TIN.from_mesh()``.

    Structural checks only -- format-specific interpretation (e.g.
    resolving LandXML ``<P>``/``<F>`` ids into these indices) is the
    caller's responsibility.
    """
    if len(vertices) == 0:
        raise TriangulationError("At least one vertex is required to build a TIN from an explicit mesh.")

    if simplices.ndim != 2 or simplices.shape[1] != 3:
        raise TriangulationError("simplices must be an (n, 3) array of triangle vertex indices.")

    if simplices.shape[0] == 0:
        raise TriangulationError("At least one triangle is required to build a TIN from an explicit mesh.")

    vertex_count = len(vertices)

    if np.any(simplices < 0) or np.any(simplices >= vertex_count):
        raise TriangulationError(
            "Triangle connectivity references a vertex index outside the range of the given vertices."
        )

    for triangle_index in range(simplices.shape[0]):
        v0, v1, v2 = (int(simplices[triangle_index, k]) for k in range(3))

        if v0 == v1 or v1 == v2 or v0 == v2:
            raise TriangulationError(f"Triangle at index {triangle_index} has duplicated vertex indices.")

        p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]

        area = _orientation(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y)

        if abs(area) <= EPSILON:
            raise TriangulationError(
                f"Triangle at index {triangle_index} is degenerate (zero XY area / collinear vertices)."
            )


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

    # PR21.5 -- uniform-grid spatial index accelerating find_triangle()/
    # contains(). Maps (row, col) grid cell -> tuple of candidate
    # triangle indices whose AABB overlaps that cell. Built lazily on
    # first use, matching the existing edges/bounds cache pattern.
    # See TIN._build_triangle_index()'s own docstring for the design
    # rationale (uniform AABB grid, not an R-tree) and correctness
    # argument (a point inside a triangle is always inside that
    # triangle's own AABB, so registering a triangle in every cell its
    # AABB overlaps can never miss a query point).
    triangle_index: dict[tuple[int, int], tuple[int, ...]] | None = None
    triangle_index_cell_size: float | None = None


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

    @classmethod
    def from_mesh(
        cls,
        vertices: tuple[Point3D, ...],
        simplices: NDArray[np.int32],
    ) -> TIN:
        """
        Build a TIN from explicit vertex/triangle connectivity,
        without recomputing a Delaunay triangulation.

        Unlike ``from_points()``, this does not call
        ``scipy.spatial.Delaunay`` at all: the triangle connectivity
        is taken exactly as given. Use this when triangle
        connectivity comes from an external source that may not
        match what a Delaunay triangulation of the same points would
        produce -- e.g. a LandXML ``<Surface>``/``<Faces>``
        definition, which can encode breaklines, trimmed boundaries,
        or manually edited TINs. Re-triangulating such a surface
        with ``from_points()`` would silently discard that
        information and produce a different, geometrically
        incorrect, surface.

        Parameters
        ----------
        vertices
            TIN vertices, in the index order referenced by
            ``simplices``.
        simplices
            Triangle vertex indices, shape ``(n, 3)``.

        Returns
        -------
        TIN

        Raises
        ------
        TriangulationError
            If ``vertices``/``simplices`` are empty, malformed,
            reference an out-of-range vertex index, or contain a
            degenerate (duplicated-vertex or zero-XY-area) triangle.
        """
        _validate_mesh(vertices, simplices)

        neighbors = _build_neighbors(simplices)

        return cls(
            DelaunayResult(
                vertices=vertices,
                simplices=simplices.copy(),
                neighbors=neighbors,
            ),
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

    def vertex_array(
        self,
    ) -> NDArray[np.float64]:
        """
        Vertex coordinates as an ``(n, 3)`` array.

        Complements ``vertices`` (a tuple of ``Point3D``, convenient
        for per-point access and for ``ContourGenerator``, which
        indexes individual vertices by triangle) with the bulk NumPy
        form that vectorized consumers need â€” currently
        ``features.terrain._mesh_utils.TINMesh``, which
        `BreaklineDetector`/`SlopeChangeDetector`/`EmbankmentDetector`
        rely on to build edge adjacency without a Python-level loop
        over `Point3D` attribute access.

        Returns
        -------
        ndarray
            XYZ coordinates for every vertex, in vertex-index order
            (row ``i`` corresponds to ``self.vertices[i]``).
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

        PR21.5: uses a uniform-grid spatial index (built lazily on
        first call, cached on the TIN's own `_TINCache` -- see
        `_build_triangle_index()`'s docstring for the design and
        correctness argument) to narrow the search to the small set
        of candidate triangles whose axis-aligned bounding box
        overlaps the query point's grid cell, then runs the exact
        same point-in-triangle test the pre-PR21 brute-force scan
        used, on just those candidates. Confirmed via PR21's own
        regression suite that this produces byte-for-byte identical
        results to the original O(triangle_count) scan for every
        query point tested (including points outside the convex
        hull, on shared triangle edges, and exactly on grid-cell
        boundaries) -- this method's contract and return value are
        unchanged; only the internal search cost improves.

        Returns
        -------
        int
            Triangle index, or -1 if (x, y) lies outside the
            triangulation's convex hull.
        """
        index, cell_size, min_x, min_y = self._build_triangle_index()

        col = int((x - min_x) // cell_size)
        row = int((y - min_y) // cell_size)

        for candidate in index.get((row, col), ()):
            p1, p2, p3 = self.triangle_vertices(candidate)

            if _point_in_triangle(x, y, p1, p2, p3):
                return candidate

        return -1

    def _build_triangle_index(
        self,
    ) -> tuple[dict[tuple[int, int], tuple[int, ...]], float, float, float]:
        """
        Build (or return the already-cached) uniform-grid spatial
        index accelerating `find_triangle()`.

        Design
        ------
        The TIN's XY bounding box is divided into a uniform grid
        with roughly one triangle per cell on average (grid side
        length ~= sqrt(triangle_count)). Each triangle is registered
        in every grid cell its own axis-aligned bounding box (AABB)
        overlaps -- a triangle spanning multiple cells is registered
        in all of them, not just one.

        This is a simple uniform AABB grid, not an R-tree or other
        external-dependency spatial structure, per an explicit
        decision made during PR21 planning: benchmark the simplest
        structure first, and only reach for something more
        sophisticated (R-tree/STRtree) if a future benchmark shows
        this isn't sufficient -- not before.

        Correctness
        -----------
        A point (x, y) that lies inside a triangle is, by definition,
        also inside that triangle's own AABB (the AABB is the
        tightest axis-aligned box containing all 3 vertices, and any
        point inside the triangle has coordinates between the min/max
        of its vertices). Therefore the grid cell containing (x, y)
        must be one of the cells that triangle's AABB overlaps, and
        therefore one of the cells that triangle was registered in.
        No containing triangle can ever be missed by only checking
        the query point's own cell.

        Returns
        -------
        tuple
            ``(index, cell_size, min_x, min_y)`` -- the cell ->
            candidate-triangle-indices mapping, the cell size, and
            the grid's origin (needed by `find_triangle()` to convert
            a query point into a cell coordinate).
        """
        if self._cache.triangle_index is not None and self._cache.triangle_index_cell_size is not None:
            min_x, min_y, _, _ = self.bounds
            return self._cache.triangle_index, self._cache.triangle_index_cell_size, min_x, min_y

        min_x, min_y, max_x, max_y = self.bounds
        triangle_count = self.triangle_count

        width = max_x - min_x
        height = max_y - min_y

        if triangle_count == 0 or width <= 0.0 or height <= 0.0:
            self._cache.triangle_index = {}
            self._cache.triangle_index_cell_size = max(width, height, 1.0)
            return self._cache.triangle_index, self._cache.triangle_index_cell_size, min_x, min_y

        grid_side = max(1, math.isqrt(triangle_count))
        cell_size = max(width, height) / grid_side

        vertices = self.vertex_array()
        simplices = self.simplices

        triangle_x = vertices[simplices, 0]
        triangle_y = vertices[simplices, 1]

        triangle_min_x = triangle_x.min(axis=1)
        triangle_max_x = triangle_x.max(axis=1)
        triangle_min_y = triangle_y.min(axis=1)
        triangle_max_y = triangle_y.max(axis=1)

        col_start = np.floor((triangle_min_x - min_x) / cell_size).astype(np.int64)
        col_end = np.floor((triangle_max_x - min_x) / cell_size).astype(np.int64)
        row_start = np.floor((triangle_min_y - min_y) / cell_size).astype(np.int64)
        row_end = np.floor((triangle_max_y - min_y) / cell_size).astype(np.int64)

        index: dict[tuple[int, int], list[int]] = {}
        for triangle_index in range(triangle_count):
            for row in range(int(row_start[triangle_index]), int(row_end[triangle_index]) + 1):
                for col in range(int(col_start[triangle_index]), int(col_end[triangle_index]) + 1):
                    index.setdefault((row, col), []).append(triangle_index)

        built_index = {key: tuple(value) for key, value in index.items()}
        self._cache.triangle_index = built_index
        self._cache.triangle_index_cell_size = cell_size

        return built_index, cell_size, min_x, min_y

    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate terrain elevation at XY coordinate.

        Uses barycentric interpolation inside the containing
        triangle.

        Parameters
        ----------
        x
            X coordinate.
        y
            Y coordinate.

        Returns
        -------
        float
            Interpolated elevation.

        Raises
        ------
        ValueError
            If the point is outside the TIN.
        """

        triangle_index = self.find_triangle(
            x,
            y,
        )

        if triangle_index < 0:
            raise ValueError(f"Point ({x}, {y}) is outside TIN.")

        p1, p2, p3 = self.triangle_vertices(
            triangle_index,
        )

        denominator = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y)

        if abs(denominator) < 1e-15:
            return float((p1.z + p2.z + p3.z) / 3.0)

        w1 = ((p2.y - p3.y) * (x - p3.x) + (p3.x - p2.x) * (y - p3.y)) / denominator

        w2 = ((p3.y - p1.y) * (x - p3.x) + (p1.x - p3.x) * (y - p3.y)) / denominator

        w3 = 1.0 - w1 - w2

        return float(w1 * p1.z + w2 * p2.z + w3 * p3.z)

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

            edge_indices.add((min(i0, i1), max(i0, i1)))
            edge_indices.add((min(i1, i2), max(i1, i2)))
            edge_indices.add((min(i2, i0), max(i2, i0)))

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
