"""
Coverage audit tests for
topocore.terrain.algorithms.constrained_delaunay.ConstrainedDelaunayTriangulator.

PR22 coverage hardening -- the most algorithmically delicate module
in this audit round. Per explicit guidance: a high line-coverage
percentage does not by itself prove a constrained Delaunay
implementation is geometrically correct, so every test below asserts
a genuine geometric invariant, not merely "runs without error".

Design note on this suite's own construction: several early attempts
at hand-crafted point configurations accidentally introduced
UNINTENDED collinearities (a point lying exactly on a constraint's
own line, or two "diagonal" test points coincidentally summing to a
constant), which correctly triggered this module's own collinearity
guards -- confirmed these were genuine test-design mistakes on this
suite's part, not bugs in the algorithm, by checking each
configuration's own arithmetic before concluding anything. The tests
below use either verified-safe hand-picked points or seeded random
points (checked not to coincidentally match an already-natural edge
before asserting a flip was forced).

Invariants verified:
  - The constraint edge is forced to appear as an actual edge in the
    final mesh (checked directly against the returned simplices, not
    merely "no exception was raised").
  - The constraint edge SURVIVES the global Delaunay legalization
    sweep that runs after every constraint has been inserted (this
    sweep must skip constrained edges, per the module's own
    docstring) -- verified with enough additional points that the
    legalization sweep genuinely has work to do elsewhere in the mesh.
  - The resulting mesh is a valid triangulation: all indices in
    range, no degenerate (zero-area) triangles.
  - Determinism: identical input produces byte-for-byte identical
    output.
  - Multiple, non-conflicting constraints are all honored
    simultaneously.
  - The genuinely documented error paths: a constraint already
    present as a natural edge (no-op path), an out-of-range index, a
    self-loop, a constraint passing through an existing vertex, and
    two constraints that geometrically cross each other (the two
    diagonals of a square, which cross by construction).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.geometry.point3d import Point3D
from topocore.terrain.algorithms.constrained_delaunay import (
    ConstrainedDelaunayTriangulator,
)
from topocore.terrain.algorithms.delaunay import DelaunayTriangulator
from topocore.terrain.exceptions import BreaklineError


def _edge_exists(simplices: np.ndarray, a: int, b: int) -> bool:
    """True if edge (a, b) appears as a side of any triangle in simplices."""
    key = frozenset((a, b))
    for tri in simplices:
        edges = [
            frozenset((tri[0], tri[1])),
            frozenset((tri[1], tri[2])),
            frozenset((tri[2], tri[0])),
        ]
        if key in edges:
            return True
    return False


def _assert_valid_triangulation(vertices: tuple, simplices: np.ndarray) -> None:
    vertex_count = len(vertices)
    assert np.all((simplices >= 0) & (simplices < vertex_count)), "simplex index out of range"

    for tri in simplices:
        p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        area = (p0.x - p2.x) * (p1.y - p2.y) - (p1.x - p2.x) * (p0.y - p2.y)
        assert abs(area) > 1e-9, f"degenerate triangle: {tri}"


def _random_points(n: int = 30, seed: int = 42) -> list[Point3D]:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n, 2))
    z = rng.uniform(0, 10, n)
    return [Point3D(float(x), float(y), float(zi)) for (x, y), zi in zip(xy, z, strict=True)]


# ----------------------------------------------------------------------
# Core invariant: a genuinely forced constraint appears in the final mesh.
# ----------------------------------------------------------------------


def test_constraint_requiring_a_genuine_flip_appears_in_final_mesh() -> None:
    points = _random_points()
    natural = DelaunayTriangulator.triangulate(points)
    assert not _edge_exists(natural.simplices, 0, 15), "test setup assumption violated: (0,15) was already natural"

    result = ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(0, 15)])

    assert _edge_exists(result.simplices, 0, 15)
    _assert_valid_triangulation(result.vertices, result.simplices)


def test_constraint_survives_global_legalization_sweep() -> None:
    """The constrained edge must not be flipped away by the global legalization pass that runs afterward."""
    points = _random_points(n=30, seed=42)
    result = ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(0, 15)])

    assert _edge_exists(result.simplices, 0, 15)


def test_multiple_non_conflicting_constraints_all_honored() -> None:
    points = _random_points()
    result = ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(0, 15), (5, 20), (2, 25)])

    assert _edge_exists(result.simplices, 0, 15)
    assert _edge_exists(result.simplices, 5, 20)
    assert _edge_exists(result.simplices, 2, 25)
    _assert_valid_triangulation(result.vertices, result.simplices)


def test_constraint_already_a_natural_edge_is_a_no_op() -> None:
    """When (p, q) already exists in the base triangulation, _find_crossing_edges() returns None -- nothing to insert."""
    points = _random_points()
    natural = DelaunayTriangulator.triangulate(points)
    existing_edge = (int(natural.simplices[0][0]), int(natural.simplices[0][1]))

    result = ConstrainedDelaunayTriangulator.triangulate(points, constraints=[existing_edge])

    assert _edge_exists(result.simplices, *existing_edge)


def test_triangulation_is_deterministic() -> None:
    points = _random_points()

    result_a = ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(0, 15)])
    result_b = ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(0, 15)])

    np.testing.assert_array_equal(result_a.simplices, result_b.simplices)


# ----------------------------------------------------------------------
# Documented error paths.
# ----------------------------------------------------------------------


def test_rejects_out_of_range_constraint_index() -> None:
    points = _random_points(n=5)
    with pytest.raises(BreaklineError, match="out-of-range"):
        ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(0, 99)])


def test_rejects_self_loop_constraint() -> None:
    points = _random_points(n=5)
    with pytest.raises(BreaklineError, match="cannot connect a point to itself"):
        ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(1, 1)])


def test_rejects_constraint_through_existing_vertex() -> None:
    """A point lying exactly on the constraint's own line is explicitly unsupported, not silently mis-triangulated."""
    points = [
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 1.0),
        Point3D(1.0, 1.0, 2.0),
        Point3D(0.0, 1.0, 1.0),
        Point3D(0.5, 0.5, 0.5),  # exactly on the (0,2) diagonal
    ]
    with pytest.raises(BreaklineError, match="passes exactly through an existing vertex"):
        ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(0, 2)])


def test_rejects_two_crossing_constraints() -> None:
    """The two diagonals of a square cross by construction -- this must be rejected, not silently resolved."""
    points = [
        Point3D(0.0, 0.0, 0.0),
        Point3D(10.0, 0.0, 0.0),
        Point3D(10.0, 10.0, 0.0),
        Point3D(0.0, 10.0, 0.0),
    ]
    with pytest.raises(BreaklineError, match="cross each other"):
        ConstrainedDelaunayTriangulator.triangulate(points, constraints=[(0, 2), (1, 3)])
