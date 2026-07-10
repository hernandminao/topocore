"""
topocore.terrain.algorithms.constrained_delaunay
====================================================

Constrained Delaunay Triangulation (CDT).

Implements Sloan's (1993) segment-insertion algorithm on top of the
unconstrained Delaunay triangulation produced by
``DelaunayTriangulator``: constraint edges (breaklines) are forced
into an existing triangulation by walking the chain of edges each
constraint crosses and repeatedly swapping diagonals until the
constraint becomes a real mesh edge, then locally restoring the
Delaunay property everywhere except along constrained edges.

This runs entirely in the XY plane (Z is carried through untouched
via point indexing), consistent with ``DelaunayTriangulator``.

Scope
-----
Handles the common case: constraint segments that do not cross each
other and do not pass exactly through an existing vertex. Both
situations are detected and raise ``BreaklineError`` rather than
producing silently incorrect geometry -- resolving them (via Steiner
point insertion) is a possible future extension, not attempted here.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final
import math
import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.terrain.algorithms.delaunay import DelaunayResult
from topocore.terrain.algorithms.delaunay import DelaunayTriangulator
from topocore.terrain.exceptions import BreaklineError
from topocore.terrain.constants import EPSILON

_XY = tuple[float, float]
_Edge = tuple[int, int]
_EdgeKey = frozenset[int]

_MAX_WALK_FACTOR: Final[int] = 4
_MAX_WALK_MARGIN: Final[int] = 64
_MAX_QUEUE_FACTOR: Final[int] = 8
_MAX_QUEUE_MARGIN: Final[int] = 64
_MAX_SWEEP_MARGIN: Final[int] = 32


# =====================================================================
# Geometric primitives
# =====================================================================


def _orient(p: _XY, q: _XY, r: _XY) -> float:
    """
    Twice the signed area of triangle (p, q, r).

    Positive when r is left of ray p->q, negative when right, zero
    when collinear.
    """
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _segments_cross(p1: _XY, p2: _XY, p3: _XY, p4: _XY) -> bool:
    """
    True if segments (p1, p2) and (p3, p4) properly intersect.

    Also doubles as the "is this quad convex" test: for a quad
    a-c-b-d, the diagonals a-b and c-d cross if and only if the quad
    is convex, which is exactly when the edge a-b may legally be
    flipped to c-d.
    """
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)

    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _in_circle(a: _XY, b: _XY, c: _XY, d: _XY) -> bool:
    """
    True if ``d`` lies strictly inside the circumcircle of (a, b, c).

    Robust to the winding order of (a, b, c): the determinant sign
    convention is normalized against the triangle's own orientation.
    """
    orientation = _orient(a, b, c)

    if abs(orientation) <= EPSILON:    
        return False

    a1, a2 = a[0] - d[0], a[1] - d[1]
    b1, b2 = b[0] - d[0], b[1] - d[1]
    c1, c2 = c[0] - d[0], c[1] - d[1]

    sq_a = a1 * a1 + a2 * a2
    sq_b = b1 * b1 + b2 * b2
    sq_c = c1 * c1 + c2 * c2

    det = (
        a1 * (b2 * sq_c - sq_b * c2)
        - a2 * (b1 * sq_c - sq_b * c1)
        + sq_a * (b1 * c2 - b2 * c1)
    )

    if orientation > 0.0:
        return det > 0.0

    return det < 0.0


# =====================================================================
# Mutable mesh: triangle list + edge -> incident-triangle map
# =====================================================================


def _build_mesh(
    simplices: NDArray[np.int32],
) -> tuple[list[list[int]], dict[_EdgeKey, list[int]]]:
    """
    Build the mutable mesh representation from scipy's simplices.
    """
    triangles = [list(map(int, triangle)) for triangle in simplices]
    edge_map: dict[_EdgeKey, list[int]] = {}

    for index, (a, b, c) in enumerate(triangles):
        for edge in (
            frozenset((a, b)),
            frozenset((b, c)),
            frozenset((c, a)),
        ):
            edge_map.setdefault(edge, []).append(index)

    return triangles, edge_map


def _apex(triangle: list[int], a: int, b: int) -> int:
    """
    Return the triangle's vertex other than ``a`` and ``b``.
    """
    for vertex in triangle:
        if vertex != a and vertex != b:
            return vertex

    raise BreaklineError(
        f"Malformed triangle {triangle} does not contain edge ({a}, {b})."
    )

def _flip_edge(
    triangles: list[list[int]],
    edge_map: dict[_EdgeKey, list[int]],
    a: int,
    b: int,
) -> _Edge:
    """
    Flip the shared edge (a, b) to the opposite diagonal (c, d).

    Returns
    -------
    tuple
        The new diagonal, ``(c, d)``.
    """
    key_ab = frozenset((a, b))
    triangle_indices = edge_map[key_ab]

    if len(triangle_indices) != 2:
        raise BreaklineError(
            f"Cannot flip a boundary edge: ({a}, {b})."
        )

    t1_idx, t2_idx = triangle_indices
    c = _apex(triangles[t1_idx], a, b)
    d = _apex(triangles[t2_idx], a, b)

    del edge_map[key_ab]

    key_bc = frozenset((b, c))
    key_ad = frozenset((a, d))

    edge_map[key_bc] = [
        t2_idx if index == t1_idx else index
        for index in edge_map[key_bc]
    ]
    edge_map[key_ad] = [
        t1_idx if index == t2_idx else index
        for index in edge_map[key_ad]
    ]

    triangles[t1_idx] = [a, c, d]
    triangles[t2_idx] = [b, c, d]

    edge_map[frozenset((c, d))] = [t1_idx, t2_idx]

    return c, d


# =====================================================================
# Locating the edges a constraint segment crosses
# =====================================================================


def _wedge_contains(
    xy: list[_XY],
    p: int,
    q: int,
    x: int,
    y: int,
) -> bool:
    """
    True if ``q`` falls inside the angular wedge at ``p`` spanned by
    rays p->x and p->y (the wedge of a triangle (p, x, y)).
    """
    pp, xx, yy, qq = xy[p], xy[x], xy[y], xy[q]

    side_q_vs_px = _orient(pp, xx, qq)
    side_y_vs_px = _orient(pp, xx, yy)

    side_q_vs_py = _orient(pp, yy, qq)
    side_x_vs_py = _orient(pp, yy, xx)

    if ( abs(side_q_vs_px) <= EPSILON or abs(side_q_vs_py) <= EPSILON ):    
        raise BreaklineError(
            "The breakline passes exactly through an existing "
            "vertex; not supported yet."
        )

    return (
        (side_q_vs_px > 0.0) == (side_y_vs_px > 0.0)
        and (side_q_vs_py > 0.0) == (side_x_vs_py > 0.0)
    )


def _find_start_edge(
    triangles: list[list[int]],
    edge_map: dict[_EdgeKey, list[int]],
    xy: list[_XY],
    p: int,
    q: int,
) -> tuple[_Edge, int] | None:
    """
    Find the edge opposite ``p``, in the triangle whose wedge at
    ``p`` contains the direction toward ``q``.
    """
    incident_triangles: set[int] = set()

    for edge, triangle_indices in edge_map.items():
        if p in edge:
            incident_triangles.update(triangle_indices)

    for triangle_index in incident_triangles:
        others = [v for v in triangles[triangle_index] if v != p]

        if len(others) != 2:
            continue

        x, y = others

        if _wedge_contains(xy, p, q, x, y):
            return (x, y), triangle_index

    return None


def _next_crossing_edge(
    xy: list[_XY],
    p: int,
    q: int,
    x: int,
    z: int,
    y: int,
) -> _Edge:
    """
    Given the crossing edge (x, y) and the apex ``z`` on its far
    side, determine which of (x, z) / (y, z) segment pq crosses next.
    """
    side_x = _orient(xy[p], xy[q], xy[x])
    side_z = _orient(xy[p], xy[q], xy[z])

    if ( abs(side_x) <= EPSILON or abs(side_z) <= EPSILON ):    
        raise BreaklineError(
            "Degenerate (collinear) configuration while walking "
            "toward the breakline endpoint."
        )

    if (side_x > 0.0) == (side_z > 0.0):
        return y, z

    return x, z


def _find_crossing_edges(
    triangles: list[list[int]],
    edge_map: dict[_EdgeKey, list[int]],
    xy: list[_XY],
    p: int,
    q: int,
) -> list[_Edge] | None:
    """
    Walk from ``p`` to ``q``, collecting every mesh edge the segment
    crosses.

    Returns
    -------
    list or None
        ``None`` if edge (p, q) already exists (nothing to insert).
    """
    if frozenset((p, q)) in edge_map:
        return None

    start = _find_start_edge(triangles, edge_map, xy, p, q)

    if start is None:
        raise BreaklineError(
            f"Could not start the crossing search between {p} and {q}."
        )

    (x, y), current_triangle = start
    crossing: list[_Edge] = []

    max_steps = len(triangles) * _MAX_WALK_FACTOR + _MAX_WALK_MARGIN

    for _ in range(max_steps):
        crossing.append((x, y))

        candidates = [
            t for t in edge_map[frozenset((x, y))] if t != current_triangle
        ]

        if not candidates:
            raise BreaklineError(
                "The breakline reached the triangulation boundary "
                "before its endpoint."
            )

        current_triangle = candidates[0]
        z = _apex(triangles[current_triangle], x, y)

        if z == q:
            return crossing

        x, y = _next_crossing_edge(xy, p, q, x, z, y)

    raise BreaklineError(
        "Exceeded the iteration limit while searching for crossing "
        "edges."
    )


# =====================================================================
# Zipping a constraint segment into the mesh
# =====================================================================


def _pop_valid_edge(
    queue: list[_Edge],
    edge_map: dict[_EdgeKey, list[int]],
    constrained: set[_EdgeKey],
) -> _Edge | None:
    """
    Pop the next queue entry, discarding entries superseded by
    earlier flips.
    """
    a, b = queue.pop(0)
    key_ab = frozenset((a, b))

    if key_ab not in edge_map:
        return None

    if key_ab in constrained:
        raise BreaklineError(
            "Two breaklines cross each other; not supported yet."
        )

    return a, b


def _try_flip_crossing_edge(
    triangles: list[list[int]],
    edge_map: dict[_EdgeKey, list[int]],
    xy: list[_XY],
    a: int,
    b: int,
    key_pq: _EdgeKey,
) -> tuple[str, _Edge | None]:
    """
    Attempt to flip crossing edge (a, b).

    Returns
    -------
    tuple
        ``("requeue", (a, b))`` if the quad isn't convex yet.
        ``("requeue", new_edge)`` if the new diagonal still crosses pq.
        ``("stop", None)`` once this edge no longer needs processing
        (either it produced the constraint edge, or its replacement
        diagonal no longer crosses pq). Delaunay legalization for any
        newly introduced diagonal is handled separately, by the
        single global sweep in ``_full_legalize``.
    """
    c = _apex(triangles[edge_map[frozenset((a, b))][0]], a, b)
    d = _apex(triangles[edge_map[frozenset((a, b))][1]], a, b)

    if not _segments_cross(xy[a], xy[b], xy[c], xy[d]):
        return "requeue", (a, b)

    new_a, new_b = _flip_edge(triangles, edge_map, a, b)

    if frozenset((new_a, new_b)) == key_pq:
        return "stop", None

    p, q = tuple(key_pq)

    if _segments_cross(xy[p], xy[q], xy[new_a], xy[new_b]):
        return "requeue", (new_a, new_b)

    return "stop", None


def _process_crossing_queue(
    triangles: list[list[int]],
    edge_map: dict[_EdgeKey, list[int]],
    xy: list[_XY],
    queue: list[_Edge],
    key_pq: _EdgeKey,
    constrained: set[_EdgeKey],
) -> None:
    """
    Zip the constraint edge into the mesh by repeatedly flipping
    crossing edges until it appears in ``edge_map``.

    Delaunay legalization is not performed here -- ``triangulate()``
    runs a single global legalization sweep after every constraint
    has been inserted, rather than after each one individually.
    """
    max_steps = len(queue) * (len(triangles) + _MAX_QUEUE_FACTOR) + _MAX_QUEUE_MARGIN
    steps = 0

    while queue:
        if key_pq in edge_map:
            # The corridor is fully zipped; leftover entries refer to
            # transient states later flips already made irrelevant.
            break

        steps += 1
        if steps > max_steps:
            raise BreaklineError(
                "Breakline insertion did not converge (the breakline "
                "may cross another breakline)."
            )

        popped = _pop_valid_edge(queue, edge_map, constrained)
        if popped is None:
            continue

        a, b = popped
        action, edge = _try_flip_crossing_edge(
            triangles, edge_map, xy, a, b, key_pq
        )

        if action == "requeue":
            queue.append(edge)


# =====================================================================
# Restoring the Delaunay property away from constrained edges
# =====================================================================


def _is_illegal(
    triangles: list[list[int]],
    xy: list[_XY],
    edge_map: dict[_EdgeKey, list[int]],
    key_ab: _EdgeKey,
) -> tuple[int, int, int, int] | None:
    """
    Return ``(a, b, c, d)`` if edge (a, b) violates the empty
    circumcircle property, else ``None``.
    """
    triangle_indices = edge_map.get(key_ab)

    if triangle_indices is None or len(triangle_indices) != 2:
        return None

    a, b = tuple(key_ab)
    c = _apex(triangles[triangle_indices[0]], a, b)
    d = _apex(triangles[triangle_indices[1]], a, b)

    if not _segments_cross(xy[a], xy[b], xy[c], xy[d]):
        return None

    if _in_circle(xy[a], xy[b], xy[c], xy[d]):
        return a, b, c, d

    return None


def _full_legalize(
    triangles: list[list[int]],
    edge_map: dict[_EdgeKey, list[int]],
    xy: list[_XY],
    constrained: set[_EdgeKey],
) -> None:
    """
    Sweep every non-constrained edge until a full pass performs zero
    flips.

    This is the correctness backstop, called once after all
    constraints have been inserted -- it does not depend on any
    single insertion having propagated its own local legalization
    perfectly. Runs in O(E) per sweep; acceptable here since Terrain
    performance work belongs to PR20, not PR12.

    Raises
    ------
    BreaklineError
        If legalization does not stabilize within a generous sweep
        budget.
    """
    max_sweeps = len(triangles) + _MAX_SWEEP_MARGIN

    for _ in range(max_sweeps):
        flipped_any = False

        for key_ab in tuple(edge_map.keys()):    
            if key_ab in constrained:
                continue

            illegal = _is_illegal(triangles, xy, edge_map, key_ab)

            if illegal is None:
                continue

            a, b, _c, _d = illegal
            _flip_edge(triangles, edge_map, a, b)
            flipped_any = True

        if not flipped_any:
            return

    raise BreaklineError("Global Delaunay legalization did not converge.")


# =====================================================================
# Public orchestration
# =====================================================================


def _insert_constraint(
    triangles: list[list[int]],
    edge_map: dict[_EdgeKey, list[int]],
    xy: list[_XY],
    p: int,
    q: int,
    constrained: set[_EdgeKey],
) -> None:
    """
    Insert a single constraint edge (p, q) into the mesh.
    """
    key_pq = frozenset((p, q))

    crossing = _find_crossing_edges(triangles, edge_map, xy, p, q)

    if crossing is None:
        constrained.add(key_pq)
        return

    _process_crossing_queue(
        triangles, edge_map, xy, list(crossing), key_pq, constrained
    )

    if key_pq not in edge_map:
        raise BreaklineError(
            f"Failed to recover constraint edge ({p}, {q})."
        )

    constrained.add(key_pq)


def _validate_constraint(
    a: int,
    b: int,
    vertex_count: int,
) -> None:
    if not (0 <= a < vertex_count) or not (0 <= b < vertex_count):
        raise BreaklineError(
            f"Constraint references an out-of-range point index: "
            f"({a}, {b})."
        )

    if a == b:
        raise BreaklineError(
            "A constraint edge cannot connect a point to itself."
        )


def _compute_neighbors(
    triangles: list[list[int]],
    edge_map: dict[_EdgeKey, list[int]],
) -> NDArray[np.int32]:
    """
    Compute scipy-convention neighbor indices: ``neighbors[i, k]`` is
    the triangle across the edge opposite vertex ``k`` of triangle
    ``i``, or -1 on the boundary.
    """
    count = len(triangles)
    neighbors = np.full((count, 3), -1, dtype=np.int32)

    for index, (v0, v1, v2) in enumerate(triangles):
        opposite_edges = ((v1, v2), (v2, v0), (v0, v1))

        for slot, (x, y) in enumerate(opposite_edges):
            others = [
                t for t in edge_map[frozenset((x, y))] if t != index
            ]
            neighbors[index, slot] = others[0] if others else -1

    return neighbors


class ConstrainedDelaunayTriangulator:
    """
    Computes a Delaunay triangulation constrained to honor a set of
    edges (breaklines).

    This class performs only the mathematical triangulation. It does
    not create TIN objects or any higher level terrain model.
    """

    __slots__ = ()

    @staticmethod
    def triangulate(
        points: list[Point3D] | tuple[Point3D, ...],
        constraints: list[_Edge] | tuple[_Edge, ...],
    ) -> DelaunayResult:
        """
        Compute a constrained Delaunay triangulation.

        Parameters
        ----------
        points
            Terrain vertices.
        constraints
            Edges to force into the triangulation, as pairs of
            indices into ``points``.

        Returns
        -------
        DelaunayResult

        Raises
        ------
        TriangulationError
            If the unconstrained base triangulation fails.
        BreaklineError
            If a constraint is malformed, crosses another constraint,
            or cannot be recovered.
        """
        base = DelaunayTriangulator.triangulate(points)
        vertices = base.vertices
        xy = [(point.x, point.y) for point in vertices]

        triangles, edge_map = _build_mesh(base.simplices)
        constrained: set[_EdgeKey] = set()

        for a, b in constraints:
            _validate_constraint(a, b, len(vertices))
            _insert_constraint(triangles, edge_map, xy, a, b, constrained)

        _full_legalize(triangles, edge_map, xy, constrained)

        return DelaunayResult(
            vertices=vertices,
            simplices=np.asarray(triangles, dtype=np.int32),
            neighbors=_compute_neighbors(triangles, edge_map),
        )


__all__ = [
    "ConstrainedDelaunayTriangulator",
]
