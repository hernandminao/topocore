"""
topocore.terrain.contours
==========================

Contour line generation from a Triangulated Irregular Network (TIN).

This module implements the marching-triangles algorithm: for every
triangle of the TIN, the elevation plane at a given level is
intersected against the triangle edges, producing short segments.
Segments are then stitched together into continuous polylines
(open or closed) to form ``ContourLine`` instances.

The algorithm operates directly on the unconstrained TIN produced by
``DelaunayTriangulator``. Breakline-aware contouring (honoring hard
and soft breaklines as triangulation constraints) will be introduced
once the constrained triangulation backend is available; until then,
contours generated over terrain with sharp breaklines may cross them.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.geometry.point3d import Point3D
from topocore.terrain.constants import DEFAULT_CONTOUR_INTERVAL, EPSILON
from topocore.terrain.exceptions import ContourError
from topocore.terrain.models import ContourLine
from topocore.terrain.tin import TIN
from topocore.terrain.validation import validate_interval

_NodeKey = tuple[int, int, int]


def _node_key(point: Point3D) -> _NodeKey:
    """
    Build a tolerance-quantized key used to weld coincident vertices.

    Contour segments produced by two triangles that share an edge
    are expected to meet at the same point, but independent
    floating-point interpolation on each side can differ by a few
    ULPs. Quantizing by ``EPSILON`` welds those endpoints together
    so segments can be stitched into polylines.
    """
    return (
        round(point.x / EPSILON),
        round(point.y / EPSILON),
        round(point.z / EPSILON),
    )


def _classify(z: float, level: float) -> int:
    """
    Classify a vertex elevation relative to a contour level.

    Returns
    -------
    int
        ``0`` if ``z`` is within ``EPSILON`` of ``level`` (ON),
        ``1`` if above, ``-1`` if below.
    """
    if abs(z - level) <= EPSILON:
        return 0

    return 1 if z > level else -1


def _triangle_segment(
    triangle: tuple[Point3D, Point3D, Point3D],
    level: float,
) -> tuple[Point3D, Point3D] | None:
    """
    Intersect a single triangle with a horizontal plane.

    Handles the degenerate case where one or two vertices sit
    exactly at ``level`` (within ``EPSILON``) explicitly, rather
    than treating it as "no crossing" -- a flat plateau/pad at
    exactly the requested contour elevation, or a slope's toe
    exactly matching it, are both real, common terrain shapes (not
    edge cases confined to synthetic geometry), and both used to
    silently disappear from contour output entirely. See the PR19
    session notes for the concrete construction-pad reproduction
    that motivated this fix.

    Returns
    -------
    tuple or None
        The two intersection points, or ``None`` if the triangle
        does not cross ``level`` (including the case where it lies
        fully on one side, or is fully coplanar with ``level`` --
        that triangle's own boundary is supplied by its neighbors,
        each of which has exactly one off-level vertex).
    """
    classes = [_classify(triangle[i].z, level) for i in range(3)]
    on_indices = [i for i in range(3) if classes[i] == 0]

    if len(on_indices) == 2:
        # The edge between the two ON vertices IS the contour
        # segment for this triangle -- no interpolation needed, and
        # none would be numerically meaningful (both endpoints are
        # already at `level`). z is normalized to exactly `level`
        # (not left as the vertex's own z, which may differ from
        # `level` by up to EPSILON) so this segment's endpoints weld
        # correctly, via `_node_key`, against interpolated points
        # from neighboring triangles that use `level` exactly.
        i, j = on_indices
        return (
            Point3D(triangle[i].x, triangle[i].y, level),
            Point3D(triangle[j].x, triangle[j].y, level),
        )

    if len(on_indices) == 3:
        # Fully coplanar with `level` -- this triangle contributes
        # no line of its own; its perimeter is supplied by whichever
        # neighboring, non-coplanar triangles share its edges (each
        # via the two-ON-vertices case above).
        return None

    if len(on_indices) == 1:
        on_index = on_indices[0]
        other_indices = [i for i in range(3) if i != on_index]
        other_classes = (classes[other_indices[0]], classes[other_indices[1]])

        if other_classes[0] == other_classes[1]:
            # The other two vertices are on the same side -- the
            # contour only touches this triangle at the single ON
            # vertex, not a real line through its interior. (Neither
            # can be ON here: that would mean 2+ ON vertices, a case
            # already handled above, before this branch is reached.)
            return None

        start, end = triangle[other_indices[0]], triangle[other_indices[1]]
        z0, z1 = start.z, end.z
        t = (level - z0) / (z1 - z0)

        crossing = Point3D(
            start.x + t * (end.x - start.x),
            start.y + t * (end.y - start.y),
            level,
        )

        return Point3D(triangle[on_index].x, triangle[on_index].y, level), crossing

    crossings: list[Point3D] = []

    for i in range(3):
        start = triangle[i]
        end = triangle[(i + 1) % 3]

        z0 = start.z
        z1 = end.z

        if (z0 - level) * (z1 - level) >= 0.0:
            continue

        t = (level - z0) / (z1 - z0)

        crossings.append(
            Point3D(
                start.x + t * (end.x - start.x),
                start.y + t * (end.y - start.y),
                level,
            )
        )

    if len(crossings) != 2:
        return None

    return crossings[0], crossings[1]


_Graph = tuple[
    dict[_NodeKey, Point3D],
    dict[_NodeKey, list[int]],
    list[tuple[_NodeKey, _NodeKey]],
]


def _build_graph(
    segments: list[tuple[Point3D, Point3D]],
) -> _Graph:
    """
    Weld raw segments into a graph of nodes and incident edges.

    Returns
    -------
    tuple
        ``(canonical, adjacency, edges)`` where ``canonical`` maps a
        welded node key to a representative ``Point3D``, ``adjacency``
        maps a node key to the indices of its incident edges, and
        ``edges`` lists every edge as a pair of node keys.
    """
    canonical: dict[_NodeKey, Point3D] = {}
    adjacency: dict[_NodeKey, list[int]] = {}
    edges: list[tuple[_NodeKey, _NodeKey]] = []

    for start, end in segments:
        key_start = _node_key(start)
        key_end = _node_key(end)

        canonical.setdefault(key_start, start)
        canonical.setdefault(key_end, end)

        edges.append((key_start, key_end))
        edge_index = len(edges) - 1

        adjacency.setdefault(key_start, []).append(edge_index)
        adjacency.setdefault(key_end, []).append(edge_index)

    return canonical, adjacency, edges


def _other_end(
    edges: list[tuple[_NodeKey, _NodeKey]],
    edge_index: int,
    node: _NodeKey,
) -> _NodeKey:
    """
    Return the endpoint of an edge other than ``node``.
    """
    start, end = edges[edge_index]
    return end if start == node else start


def _next_unvisited_edge(
    adjacency: dict[_NodeKey, list[int]],
    visited: list[bool],
    node: _NodeKey,
) -> int | None:
    """
    Return an unvisited edge incident to ``node``, if any remain.
    """
    for edge_index in adjacency[node]:
        if not visited[edge_index]:
            return edge_index

    return None


def _walk(
    edges: list[tuple[_NodeKey, _NodeKey]],
    adjacency: dict[_NodeKey, list[int]],
    visited: list[bool],
    start_node: _NodeKey,
) -> list[_NodeKey]:
    """
    Follow unvisited edges from ``start_node``.

    Stops when no unvisited edge remains, or when the walk loops
    back to ``start_node`` (a closed contour ring).
    """
    chain = [start_node]
    current = start_node

    while True:
        edge_index = _next_unvisited_edge(adjacency, visited, current)

        if edge_index is None:
            break

        visited[edge_index] = True
        current = _other_end(edges, edge_index, current)
        chain.append(current)

        if current == start_node:
            break

    return chain


def _finalize_chain(
    canonical: dict[_NodeKey, Point3D],
    chain: list[_NodeKey],
) -> tuple[tuple[Point3D, ...], bool]:
    """
    Convert a chain of node keys into a ``(points, closed)`` pair.
    """
    points = tuple(canonical[key] for key in chain)
    return points, chain[0] == chain[-1]


def _open_polylines(
    graph: _Graph,
    visited: list[bool],
) -> list[tuple[tuple[Point3D, ...], bool]]:
    """
    Trace polylines starting at nodes with a single incident edge.
    """
    canonical, adjacency, edges = graph
    polylines: list[tuple[tuple[Point3D, ...], bool]] = []

    for node, incident in adjacency.items():
        if len(incident) != 1 or visited[incident[0]]:
            continue

        chain = _walk(edges, adjacency, visited, node)
        polylines.append(_finalize_chain(canonical, chain))

    return polylines


def _closed_polylines(
    graph: _Graph,
    visited: list[bool],
) -> list[tuple[tuple[Point3D, ...], bool]]:
    """
    Trace the closed loops left over after ``_open_polylines``.
    """
    canonical, adjacency, edges = graph
    polylines: list[tuple[tuple[Point3D, ...], bool]] = []

    for edge_index, done in enumerate(visited):
        if done:
            continue

        chain = _walk(edges, adjacency, visited, edges[edge_index][0])

        if len(chain) < 2:
            continue

        polylines.append(_finalize_chain(canonical, chain))

    return polylines


def _stitch(
    segments: list[tuple[Point3D, Point3D]],
) -> list[tuple[tuple[Point3D, ...], bool]]:
    """
    Merge unordered segments into continuous polylines.

    Parameters
    ----------
    segments
        Disconnected two-point segments produced by
        ``_triangle_segment``.

    Returns
    -------
    list
        Pairs of ``(points, closed)`` describing each polyline.

    Notes
    -----
    Split into ``_build_graph`` / ``_walk`` / ``_open_polylines`` /
    ``_closed_polylines`` so each step stays simple on its own
    (SonarQube python:S3776 -- Cognitive Complexity).
    """
    graph = _build_graph(segments)
    visited = [False] * len(graph[2])

    polylines = _open_polylines(graph, visited)
    polylines.extend(_closed_polylines(graph, visited))

    return polylines


def _levels(
    min_z: float,
    max_z: float,
    interval: float,
    base: float,
) -> list[float]:
    """
    Compute the elevation levels to contour within a range.
    """
    first = base + math.ceil((min_z - base) / interval) * interval

    levels: list[float] = []
    level = first

    while level <= max_z + EPSILON:
        levels.append(level)
        level += interval

    return levels


class ContourGenerator:
    """
    Generates contour lines from a TIN using marching triangles.
    """

    __slots__ = ("_tin",)

    def __init__(
        self,
        tin: TIN,
    ) -> None:
        self._tin = tin

    @property
    def tin(
        self,
    ) -> TIN:
        return self._tin

    def generate(
        self,
        interval: float = DEFAULT_CONTOUR_INTERVAL,
        *,
        base: float = 0.0,
    ) -> tuple[ContourLine, ...]:
        """
        Generate all contour lines at a regular elevation interval.

        Parameters
        ----------
        interval
            Vertical spacing between consecutive contour levels.
        base
            Elevation offset used to align the levels.

        Raises
        ------
        ContourError
            If the TIN has no vertices.
        """
        validate_interval(interval)

        vertices = self._tin.vertices

        if not vertices:
            raise ContourError("Cannot generate contours from an empty TIN.")

        elevations = [vertex.z for vertex in vertices]

        levels = _levels(
            min(elevations),
            max(elevations),
            interval,
            base,
        )

        contours: list[ContourLine] = []

        for level in levels:
            contours.extend(self.generate_at(level))

        return tuple(contours)

    def generate_at(
        self,
        elevation: float,
    ) -> tuple[ContourLine, ...]:
        """
        Generate the contour line(s) at a single elevation.
        """
        vertices = self._tin.vertices
        segments: list[tuple[Point3D, Point3D]] = []

        for simplex in self._tin.simplices:
            triangle = (
                vertices[int(simplex[0])],
                vertices[int(simplex[1])],
                vertices[int(simplex[2])],
            )

            segment = _triangle_segment(triangle, elevation)

            if segment is not None:
                segments.append(segment)

        return tuple(
            ContourLine(
                elevation=elevation,
                points=points,
                closed=closed,
            )
            for points, closed in _stitch(segments)
            if len(points) >= 2
        )

    def __call__(
        self,
        interval: float = DEFAULT_CONTOUR_INTERVAL,
        *,
        base: float = 0.0,
    ) -> tuple[ContourLine, ...]:
        return self.generate(
            interval,
            base=base,
        )


__all__ = [
    "ContourGenerator",
]
