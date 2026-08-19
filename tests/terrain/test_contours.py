"""
Regression suite for topocore.terrain.contours -- PR19.

Includes a real, CRITICAL bug found and fixed in this session: when
a contour level exactly matches vertex elevations (a flat pad/plateau
or a graded slope's toe at a round design elevation -- a common real
construction-topography shape, not a synthetic corner case), the
contour at that level disappeared entirely. Confirmed first with a
5-vertex synthetic pyramid, then reproduced with a realistic 25-vertex
Delaunay-triangulated graded pad (see session notes) before being
classified CRITICAL and fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.contours import ContourGenerator
from topocore.terrain.exceptions import ContourError
from topocore.terrain.tin import TIN


@pytest.fixture
def pyramid_tin() -> TIN:
    """
    4 base corners at z=0, single center peak at z=10 -- the
    original synthetic reproduction of the bug.
    """
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(2.0, 0.0, 0.0),
        Point3D(2.0, 2.0, 0.0),
        Point3D(0.0, 2.0, 0.0),
        Point3D(1.0, 1.0, 10.0),
    )
    simplices = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    return TIN.from_mesh(vertices, simplices)


@pytest.fixture
def graded_pad_tin() -> TIN:
    """
    A 5x5 grid (real Delaunay triangulation via TIN.from_points(),
    not hand-picked simplices): the center 3x3 is a flat pad at
    z=100.0, the outer ring slopes down to z=90.0 -- a realistic
    graded building pad, the exact shape that motivated classifying
    this bug as CRITICAL rather than a synthetic edge case.
    """
    points = []
    for row in range(5):
        for col in range(5):
            is_pad = 1 <= row <= 3 and 1 <= col <= 3
            z = 100.0 if is_pad else 90.0
            points.append(Point3D(float(col), float(row), z))
    return TIN.from_points(tuple(points))


# ----------------------------------------------------------------------
# Control cases: normal, non-degenerate levels -- must be unaffected
# by the fix (regression guard).
# ----------------------------------------------------------------------


def test_pyramid_control_level_unaffected_by_fix(pyramid_tin: TIN) -> None:
    """
    Level=5.0 (halfway up the pyramid) -- verified geometry: a
    closed square ring at half the base-to-center distance from
    each edge, exactly as it was before this fix.
    """
    contours = ContourGenerator(pyramid_tin).generate_at(5.0)

    assert len(contours) == 1
    contour = contours[0]
    assert contour.closed is True
    assert contour.vertex_count == 5  # 4 corners + closing point

    xs = sorted({round(p.x, 6) for p in contour.points})
    ys = sorted({round(p.y, 6) for p in contour.points})
    assert xs == [0.5, 1.5]
    assert ys == [0.5, 1.5]


def test_graded_pad_control_level_unaffected_by_fix(graded_pad_tin: TIN) -> None:
    contours = ContourGenerator(graded_pad_tin).generate_at(95.0)

    assert len(contours) == 1
    assert contours[0].closed is True
    assert contours[0].vertex_count == 21


# ----------------------------------------------------------------------
# The bug: level exactly at vertex elevations.
# ----------------------------------------------------------------------


def test_pyramid_base_exact_elevation_produces_closed_contour(pyramid_tin: TIN) -> None:
    """
    Before the fix: 0 contours at level=0.0 (exactly the 4 base
    vertices' elevation), even though the base perimeter is a real,
    valid square contour.
    """
    contours = ContourGenerator(pyramid_tin).generate_at(0.0)

    assert len(contours) == 1
    contour = contours[0]
    assert contour.closed is True
    assert contour.vertex_count == 5

    xs = sorted({round(p.x, 6) for p in contour.points})
    ys = sorted({round(p.y, 6) for p in contour.points})
    assert xs == [0.0, 2.0]
    assert ys == [0.0, 2.0]


def test_pad_exact_elevation_produces_closed_contour(graded_pad_tin: TIN) -> None:
    """
    The CRITICAL reproduction: a flat pad's own design elevation
    (100.0, exactly matching 9 of the 25 vertices) must produce its
    boundary as a real contour, not disappear.
    """
    contours = ContourGenerator(graded_pad_tin).generate_at(100.0)

    assert len(contours) == 1
    assert contours[0].closed is True
    assert contours[0].vertex_count == 9  # 8 boundary points of the 3x3 pad + closing point


def test_outer_ring_exact_elevation_produces_closed_contour(
    graded_pad_tin: TIN,
) -> None:
    """
    The graded ring's own elevation (90.0, matching the outer 16
    vertices) must also produce its boundary, not disappear.
    """
    contours = ContourGenerator(graded_pad_tin).generate_at(90.0)

    assert len(contours) == 1
    assert contours[0].closed is True
    assert contours[0].vertex_count == 13


def test_full_generate_includes_exact_elevation_levels() -> None:
    """
    generate() (the auto-level-from-interval entry point) must not
    silently drop a level just because it happens to land exactly on
    vertex elevations -- confirmed with base=0.0, interval=2.5 over
    a 0-10 pyramid, where level=0.0 (the base) is a legitimate first
    level to include.
    """
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(2.0, 0.0, 0.0),
        Point3D(2.0, 2.0, 0.0),
        Point3D(0.0, 2.0, 0.0),
        Point3D(1.0, 1.0, 10.0),
    )
    simplices = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    tin = TIN.from_mesh(vertices, simplices)

    contours = ContourGenerator(tin).generate(interval=2.5, base=0.0)
    levels = sorted({c.elevation for c in contours})

    assert 0.0 in levels


# ----------------------------------------------------------------------
# Genuinely degenerate case, unaffected: a single vertex exactly at
# the level, with both OTHER vertices on the SAME side -- the
# contour only touches at a point, not a real line. Must still be
# excluded (not a regression of this fix, a deliberately preserved
# behavior).
# ----------------------------------------------------------------------


def test_single_vertex_touch_produces_no_contour(pyramid_tin: TIN) -> None:
    """
    Level=10.0 -- only the single peak vertex is at that elevation;
    every triangle's other two vertices are both below it. No real
    line passes through any triangle's interior at this level.
    """
    contours = ContourGenerator(pyramid_tin).generate_at(10.0)
    assert len(contours) == 0


# ----------------------------------------------------------------------
# Out-of-range / basic validation, unaffected by the fix.
# ----------------------------------------------------------------------


def test_level_outside_range_produces_no_contours(pyramid_tin: TIN) -> None:
    assert len(ContourGenerator(pyramid_tin).generate_at(20.0)) == 0
    assert len(ContourGenerator(pyramid_tin).generate_at(-5.0)) == 0


def test_empty_tin_raises_contour_error() -> None:
    # TIN.from_mesh already rejects an empty vertex tuple at
    # construction (verified in Entrega 1 of PR18C) -- ContourError
    # can only realistically be reached if a TIN were ever
    # constructed with zero vertices some other way, but the
    # generator's own guard is still tested directly here (via a
    # minimal stub) in case that invariant ever changes.
    class _EmptyTinStub:
        vertices: tuple[Point3D, ...] = ()

    with pytest.raises(ContourError):
        ContourGenerator(_EmptyTinStub()).generate()  # type: ignore[arg-type]
