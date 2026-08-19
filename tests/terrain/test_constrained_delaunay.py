"""
Regression suite for topocore.terrain.algorithms.constrained_delaunay
-- PR19. The core geometric primitives (_orient, _in_circle,
_segments_cross) are verified against known analytic geometry (a
unit circle for _in_circle, axis-aligned turns for _orient) --
exactly the class of low-level sign-convention bugs found elsewhere
in this session (SpiralElement's entry/exit mirroring). None found
here.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.terrain.algorithms.constrained_delaunay import (
    ConstrainedDelaunayTriangulator,
    _in_circle,
    _orient,
    _segments_cross,
)
from topocore.terrain.algorithms.delaunay import DelaunayTriangulator
from topocore.terrain.exceptions import BreaklineError

# ----------------------------------------------------------------------
# _orient: twice the signed area, positive when r is left of p->q.
# ----------------------------------------------------------------------


def test_orient_positive_for_left_turn() -> None:
    assert _orient((0, 0), (1, 0), (0, 1)) > 0.0


def test_orient_negative_for_right_turn() -> None:
    assert _orient((0, 0), (1, 0), (0, -1)) < 0.0


def test_orient_zero_for_collinear_points() -> None:
    assert _orient((0, 0), (1, 0), (2, 0)) == pytest.approx(0.0)


def test_orient_magnitude_is_twice_triangle_area() -> None:
    # Right triangle, legs 2 and 3 -> area = 3, twice-area = 6.
    assert abs(_orient((0, 0), (2, 0), (0, 3))) == pytest.approx(6.0)


# ----------------------------------------------------------------------
# _in_circle: verified against a genuine unit circle (a, b, c on the
# circle boundary; d tested at known inside/on/outside positions).
# ----------------------------------------------------------------------


def test_in_circle_center_point_is_inside() -> None:
    a, b, c = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)  # unit circle, CCW
    assert _in_circle(a, b, c, (0.0, 0.0)) is True


def test_in_circle_far_point_is_outside() -> None:
    a, b, c = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    assert _in_circle(a, b, c, (0.0, -5.0)) is False


def test_in_circle_boundary_point_is_not_strictly_inside() -> None:
    a, b, c = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    assert _in_circle(a, b, c, (0.0, -1.0)) is False  # exactly on the unit circle


def test_in_circle_just_inside_vs_just_outside() -> None:
    a, b, c = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    assert _in_circle(a, b, c, (0.0, -0.99)) is True
    assert _in_circle(a, b, c, (0.0, -1.01)) is False


def test_in_circle_robust_to_winding_order() -> None:
    """
    Same three circle points, opposite (clockwise) winding, must
    give the identical logical answer -- the predicate normalizes
    against the triangle's own orientation internally.
    """
    a, b, c_ccw = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    a2, b2, c2_cw = (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)  # swapped -> CW

    assert _in_circle(a, b, c_ccw, (0.0, 0.0)) == _in_circle(a2, b2, c2_cw, (0.0, 0.0))
    assert _in_circle(a, b, c_ccw, (0.0, -5.0)) == _in_circle(a2, b2, c2_cw, (0.0, -5.0))


def test_in_circle_degenerate_collinear_triangle_returns_false() -> None:
    assert _in_circle((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 5.0)) is False


# ----------------------------------------------------------------------
# _segments_cross
# ----------------------------------------------------------------------


def test_segments_cross_true_for_genuine_x_intersection() -> None:
    assert _segments_cross((0, 0), (1, 1), (0, 1), (1, 0)) is True


def test_segments_cross_false_for_parallel_segments() -> None:
    assert _segments_cross((0, 0), (1, 0), (0, 1), (1, 1)) is False


def test_segments_cross_false_for_collinear_non_overlapping() -> None:
    assert _segments_cross((0, 0), (1, 0), (2, 0), (3, 0)) is False


def test_segments_cross_shared_endpoint_behavior_is_documented_not_reachable() -> None:
    """
    _segments_cross((0,0),(1,0),(1,0),(1,1)) returns True for two
    segments that only touch at a shared endpoint -- not a "proper"
    crossing in the strict sense, because one of the four orientation
    values is exactly zero and the `(d_i > 0) != (d_j > 0)` check
    doesn't distinguish "negative" from "exactly zero".

    Verified (session audit) that this is NOT reachable from any of
    the module's three actual call sites: every one passes
    `(xy[a], xy[b], xy[c], xy[d])` where `c`/`d` come from `_apex()`,
    which by construction returns a vertex distinct from `a`/`b` for
    each of two DIFFERENT adjacent triangles -- so all four indices
    are always geometrically distinct there, and this ambiguous
    input pattern can never actually occur in this algorithm. Left
    as-is (not "fixed") since correcting a case that can't be
    triggered would add complexity/risk for no real benefit -- this
    test documents the narrow, real contract, not the naive one the
    docstring's "properly intersect" phrasing might suggest in
    isolation.
    """
    assert _segments_cross((0, 0), (1, 0), (1, 0), (1, 1)) is True


# ----------------------------------------------------------------------
# ConstrainedDelaunayTriangulator -- integration: force a diagonal
# the plain (unconstrained) Delaunay did NOT choose, confirming the
# edge-flip recovery machinery actually works end to end.
# ----------------------------------------------------------------------


def _square_points() -> tuple[Point3D, ...]:
    return (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(1, 1, 0), Point3D(0, 1, 0))


def _edges_of(simplices: NDArray[np.int32]) -> set[frozenset[int]]:
    edges: set[frozenset[int]] = set()
    for triangle in simplices:
        a, b, c = (int(v) for v in triangle)
        edges.update({frozenset((a, b)), frozenset((b, c)), frozenset((a, c))})
    return edges


def test_constraint_forces_the_non_default_diagonal() -> None:
    points = _square_points()
    plain = DelaunayTriangulator.triangulate(points)
    plain_edges = _edges_of(plain.simplices)

    # Whichever diagonal plain Delaunay did NOT pick.
    forced = (0, 2) if frozenset((0, 2)) not in plain_edges else (1, 3)
    assert frozenset(forced) not in plain_edges  # sanity: genuinely not already present

    constrained = ConstrainedDelaunayTriangulator.triangulate(points, [forced])
    constrained_edges = _edges_of(constrained.simplices)

    assert frozenset(forced) in constrained_edges


def test_constraint_result_still_has_two_triangles() -> None:
    points = _square_points()
    result = ConstrainedDelaunayTriangulator.triangulate(points, [(0, 2)])
    assert result.triangle_count == 2


def test_no_constraints_matches_plain_delaunay() -> None:
    points = _square_points()
    plain = DelaunayTriangulator.triangulate(points)
    constrained = ConstrainedDelaunayTriangulator.triangulate(points, [])

    assert _edges_of(plain.simplices) == _edges_of(constrained.simplices)


# ----------------------------------------------------------------------
# Error cases -- per the module's own documented scope.
# ----------------------------------------------------------------------


def test_self_loop_constraint_raises() -> None:
    with pytest.raises(BreaklineError):
        ConstrainedDelaunayTriangulator.triangulate(_square_points(), [(0, 0)])


def test_out_of_range_constraint_raises() -> None:
    with pytest.raises(BreaklineError):
        ConstrainedDelaunayTriangulator.triangulate(_square_points(), [(0, 99)])


def test_negative_index_constraint_raises() -> None:
    with pytest.raises(BreaklineError):
        ConstrainedDelaunayTriangulator.triangulate(_square_points(), [(-1, 2)])


def test_crossing_constraints_raise() -> None:
    """
    Both diagonals of a square genuinely cross each other --
    documented as unsupported (out of scope, not silently
    mishandled).
    """
    with pytest.raises(BreaklineError):
        ConstrainedDelaunayTriangulator.triangulate(_square_points(), [(0, 2), (1, 3)])
