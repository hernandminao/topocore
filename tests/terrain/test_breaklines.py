"""
Regression suite for topocore.terrain.breaklines -- PR19.
"""

from __future__ import annotations

from topocore.geometry.point3d import Point3D
from topocore.terrain.algorithms.delaunay import DelaunayTriangulator
from topocore.terrain.breaklines import BreaklineEnforcer, apply_breaklines
from topocore.terrain.enums import BreaklineType
from topocore.terrain.exceptions import BreaklineError
from topocore.terrain.models import Breakline
from topocore.terrain.tin import TIN

import pytest


def _edges_of(tin: TIN) -> set[frozenset[int]]:
    edges: set[frozenset[int]] = set()
    for i in range(tin.triangle_count):
        a, b, c = (int(v) for v in tin.simplices[i])
        edges.update({frozenset((a, b)), frozenset((b, c)), frozenset((a, c))})
    return edges


def _square_tin() -> TIN:
    return TIN.from_points((Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(1, 1, 0), Point3D(0, 1, 0)))


# ----------------------------------------------------------------------
# Empty breaklines -- identity, no triangulation performed.
# ----------------------------------------------------------------------


def test_empty_breaklines_returns_same_tin_object() -> None:
    tin = _square_tin()
    assert apply_breaklines(tin, []) is tin


# ----------------------------------------------------------------------
# HARD breakline forces its edge into the triangulation.
# ----------------------------------------------------------------------


def test_hard_breakline_forces_the_non_default_diagonal() -> None:
    tin = _square_tin()
    plain_edges = _edges_of(tin)

    forced_indices = (0, 2) if frozenset((0, 2)) not in plain_edges else (1, 3)
    assert frozenset(forced_indices) not in plain_edges  # sanity

    points = tin.vertices
    breakline = Breakline(
        id="b1",
        name="forced diagonal",
        points=(points[forced_indices[0]], points[forced_indices[1]]),
        type=BreaklineType.HARD,
    )

    result = apply_breaklines(tin, [breakline])

    assert frozenset(forced_indices) in _edges_of(result)


def test_hard_breakline_does_not_duplicate_existing_vertices() -> None:
    tin = _square_tin()
    points = tin.vertices

    breakline = Breakline(id="b1", name="edge", points=(points[0], points[1]), type=BreaklineType.HARD)
    result = apply_breaklines(tin, [breakline])

    assert result.vertex_count == tin.vertex_count  # both points already existed


# ----------------------------------------------------------------------
# SOFT breakline: adds vertices, does not necessarily force an edge.
# ----------------------------------------------------------------------


def test_soft_breakline_adds_new_vertices() -> None:
    tin = _square_tin()
    new_points = (Point3D(0.5, 0.0, 99.0), Point3D(0.5, 1.0, 99.0))
    breakline = Breakline(id="b2", name="mid", points=new_points, type=BreaklineType.SOFT)

    result = apply_breaklines(tin, [breakline])

    assert result.vertex_count == tin.vertex_count + 2


def test_soft_breakline_preserves_added_vertex_elevation() -> None:
    tin = _square_tin()
    new_points = (Point3D(0.5, 0.0, 42.0), Point3D(0.5, 1.0, 42.0))
    breakline = Breakline(id="b2", name="mid", points=new_points, type=BreaklineType.SOFT)

    result = apply_breaklines(tin, [breakline])

    elevations = {p.z for p in result.vertices}
    assert 42.0 in elevations


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_breakline_with_fewer_than_two_points_raises() -> None:
    tin = _square_tin()
    breakline = Breakline(id="bad", name="too short", points=(Point3D(0, 0, 0),))

    with pytest.raises(BreaklineError, match="at least 2 points"):
        apply_breaklines(tin, [breakline])


def test_apply_breaklines_does_not_mutate_original_tin() -> None:
    tin = _square_tin()
    original_vertex_count = tin.vertex_count

    apply_breaklines(tin, [Breakline(id="b2", name="mid", points=(Point3D(0.5, 0.0, 99.0), Point3D(0.5, 1.0, 99.0)))])

    assert tin.vertex_count == original_vertex_count


# ----------------------------------------------------------------------
# BreaklineEnforcer class wrapper
# ----------------------------------------------------------------------


def test_breakline_enforcer_apply_matches_function() -> None:
    tin = _square_tin()
    breakline = Breakline(id="b2", name="mid", points=(Point3D(0.5, 0.0, 99.0), Point3D(0.5, 1.0, 99.0)))

    enforcer = BreaklineEnforcer(tin, [breakline])
    via_class = enforcer.apply()
    via_function = apply_breaklines(tin, [breakline])

    assert via_class.vertex_count == via_function.vertex_count


def test_breakline_enforcer_callable_matches_apply() -> None:
    tin = _square_tin()
    breakline = Breakline(id="b2", name="mid", points=(Point3D(0.5, 0.0, 99.0), Point3D(0.5, 1.0, 99.0)))
    enforcer = BreaklineEnforcer(tin, [breakline])

    assert enforcer().vertex_count == enforcer.apply().vertex_count


def test_breakline_enforcer_exposes_tin_and_breaklines() -> None:
    tin = _square_tin()
    breakline = Breakline(id="b2", name="mid", points=(Point3D(0.5, 0.0, 99.0), Point3D(0.5, 1.0, 99.0)))
    enforcer = BreaklineEnforcer(tin, [breakline])

    assert enforcer.tin is tin
    assert enforcer.breaklines == (breakline,)
