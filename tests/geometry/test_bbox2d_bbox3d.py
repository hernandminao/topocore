"""
Regression/coverage suite for topocore.geometry.bbox2d.BBox2D and
.bbox3d.BBox3D -- PR20 coverage phase.

Verified with known dimensions/area/volume, the inclusive-boundary
convention shared consistently between contains() and intersects()
(a point/box exactly ON the boundary counts as inside/intersecting
-- confirmed both classes agree on this), degenerate zero-width/
zero-volume boxes (valid, not rejected), expand()/union() including
the margin=0 no-op case, and coordinate/ordering validation (NaN,
infinity, min > max all rejected at construction). No bugs found --
both classes were already correct and mutually consistent; only
test coverage was added. (Neither class has a from_points() factory
constructor -- confirmed this is genuinely absent from the API, not
a coverage gap to test around.)
"""

from __future__ import annotations

import pytest

from topocore.core.exceptions import TopoCoreError
from topocore.geometry.bbox2d import BBox2D
from topocore.geometry.bbox3d import BBox3D
from topocore.geometry.point2d import Point2D
from topocore.geometry.point3d import Point3D

# ----------------------------------------------------------------------
# BBox2D -- dimensions, center, corners.
# ----------------------------------------------------------------------


def test_bbox2d_dimensions_and_area() -> None:
    box = BBox2D(0.0, 0.0, 10.0, 5.0)
    assert box.width == pytest.approx(10.0)
    assert box.height == pytest.approx(5.0)
    assert box.area == pytest.approx(50.0)


def test_bbox2d_center_and_corners() -> None:
    box = BBox2D(0.0, 0.0, 10.0, 5.0)
    assert box.center == Point2D(5.0, 2.5)
    assert box.min_point == Point2D(0.0, 0.0)
    assert box.max_point == Point2D(10.0, 5.0)


def test_bbox2d_degenerate_zero_width_is_valid() -> None:
    line_box = BBox2D(5.0, 0.0, 5.0, 10.0)
    assert line_box.width == 0.0
    assert line_box.area == 0.0


# ----------------------------------------------------------------------
# BBox2D -- contains, inclusive boundary.
# ----------------------------------------------------------------------


def test_bbox2d_contains_interior_point() -> None:
    box = BBox2D(0.0, 0.0, 10.0, 5.0)
    assert box.contains(Point2D(5.0, 2.5)) is True


def test_bbox2d_contains_boundary_point_is_inclusive() -> None:
    box = BBox2D(0.0, 0.0, 10.0, 5.0)
    assert box.contains(Point2D(10.0, 5.0)) is True
    assert box.contains(Point2D(0.0, 0.0)) is True


def test_bbox2d_contains_exterior_point() -> None:
    box = BBox2D(0.0, 0.0, 10.0, 5.0)
    assert box.contains(Point2D(11.0, 2.5)) is False


# ----------------------------------------------------------------------
# BBox2D -- intersects, inclusive boundary (touching counts).
# ----------------------------------------------------------------------


def test_bbox2d_intersects_overlapping() -> None:
    box1 = BBox2D(0.0, 0.0, 10.0, 5.0)
    box2 = BBox2D(5.0, 0.0, 15.0, 5.0)
    assert box1.intersects(box2) is True


def test_bbox2d_intersects_touching_exactly_is_inclusive() -> None:
    """Consistent with contains()'s own inclusive <= bounds."""
    box1 = BBox2D(0.0, 0.0, 10.0, 5.0)
    box2 = BBox2D(10.0, 0.0, 20.0, 5.0)
    assert box1.intersects(box2) is True


def test_bbox2d_intersects_separate() -> None:
    box1 = BBox2D(0.0, 0.0, 10.0, 5.0)
    box2 = BBox2D(20.0, 0.0, 30.0, 5.0)
    assert box1.intersects(box2) is False


# ----------------------------------------------------------------------
# BBox2D -- expand, union.
# ----------------------------------------------------------------------


def test_bbox2d_expand() -> None:
    box = BBox2D(0.0, 0.0, 10.0, 5.0)
    expanded = box.expand(2.0)
    assert expanded == BBox2D(-2.0, -2.0, 12.0, 7.0)


def test_bbox2d_expand_zero_margin_is_noop() -> None:
    box = BBox2D(0.0, 0.0, 10.0, 5.0)
    assert box.expand(0.0) == box


def test_bbox2d_expand_rejects_negative_margin() -> None:
    box = BBox2D(0.0, 0.0, 10.0, 5.0)
    with pytest.raises(ValueError, match="non-negative"):
        box.expand(-1.0)


def test_bbox2d_union() -> None:
    box1 = BBox2D(0.0, 0.0, 10.0, 5.0)
    box2 = BBox2D(20.0, 0.0, 30.0, 5.0)
    assert box1.union(box2) == BBox2D(0.0, 0.0, 30.0, 5.0)


def test_bbox2d_union_with_overlapping_box() -> None:
    box1 = BBox2D(0.0, 0.0, 10.0, 10.0)
    box2 = BBox2D(5.0, -5.0, 15.0, 5.0)
    assert box1.union(box2) == BBox2D(0.0, -5.0, 15.0, 10.0)


# ----------------------------------------------------------------------
# BBox2D -- to_dict, validation.
# ----------------------------------------------------------------------


def test_bbox2d_to_dict() -> None:
    box = BBox2D(0.0, 1.0, 10.0, 5.0)
    assert box.to_dict() == {"min_x": 0.0, "min_y": 1.0, "max_x": 10.0, "max_y": 5.0}


def test_bbox2d_rejects_min_greater_than_max() -> None:
    with pytest.raises(ValueError, match="min_x"):
        BBox2D(10.0, 0.0, 5.0, 5.0)
    with pytest.raises(ValueError, match="min_y"):
        BBox2D(0.0, 10.0, 5.0, 5.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_bbox2d_rejects_non_finite_coordinates(bad_value: float) -> None:
    with pytest.raises(TopoCoreError):
        BBox2D(bad_value, 0.0, 10.0, 5.0)


# ----------------------------------------------------------------------
# BBox3D -- dimensions, volume, center, corners.
# ----------------------------------------------------------------------


def test_bbox3d_dimensions_and_volume() -> None:
    box = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    assert box.width == pytest.approx(10.0)
    assert box.height == pytest.approx(5.0)
    assert box.depth == pytest.approx(2.0)
    assert box.volume == pytest.approx(100.0)


def test_bbox3d_center_and_corners() -> None:
    box = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    assert box.center == Point3D(5.0, 2.5, 1.0)
    assert box.min_point == Point3D(0.0, 0.0, 0.0)
    assert box.max_point == Point3D(10.0, 5.0, 2.0)


def test_bbox3d_degenerate_point_box_is_valid() -> None:
    point_box = BBox3D(5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert point_box.volume == 0.0


# ----------------------------------------------------------------------
# BBox3D -- contains, inclusive boundary.
# ----------------------------------------------------------------------


def test_bbox3d_contains_interior_point() -> None:
    box = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    assert box.contains(Point3D(5.0, 2.5, 1.0)) is True


def test_bbox3d_contains_boundary_point_is_inclusive() -> None:
    box = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    assert box.contains(Point3D(10.0, 5.0, 2.0)) is True


def test_bbox3d_contains_exterior_point_in_z() -> None:
    box = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    assert box.contains(Point3D(5.0, 2.5, 3.0)) is False


# ----------------------------------------------------------------------
# BBox3D -- intersects, inclusive boundary.
# ----------------------------------------------------------------------


def test_bbox3d_intersects_touching_exactly_is_inclusive() -> None:
    box1 = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    box2 = BBox3D(10.0, 0.0, 0.0, 20.0, 5.0, 2.0)
    assert box1.intersects(box2) is True


def test_bbox3d_intersects_separate_in_z() -> None:
    box1 = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    box2 = BBox3D(0.0, 0.0, 5.0, 10.0, 5.0, 7.0)
    assert box1.intersects(box2) is False


# ----------------------------------------------------------------------
# BBox3D -- expand, union.
# ----------------------------------------------------------------------


def test_bbox3d_expand() -> None:
    box = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    expanded = box.expand(1.0)
    assert expanded == BBox3D(-1.0, -1.0, -1.0, 11.0, 6.0, 3.0)


def test_bbox3d_expand_zero_margin_is_noop() -> None:
    box = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    assert box.expand(0.0) == box


def test_bbox3d_expand_rejects_negative_margin() -> None:
    box = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        box.expand(-1.0)


def test_bbox3d_union() -> None:
    box1 = BBox3D(0.0, 0.0, 0.0, 10.0, 5.0, 2.0)
    box2 = BBox3D(20.0, 0.0, 0.0, 30.0, 5.0, 2.0)
    assert box1.union(box2) == BBox3D(0.0, 0.0, 0.0, 30.0, 5.0, 2.0)


# ----------------------------------------------------------------------
# BBox3D -- to_dict, validation.
# ----------------------------------------------------------------------


def test_bbox3d_to_dict() -> None:
    box = BBox3D(0.0, 1.0, 2.0, 10.0, 5.0, 8.0)
    assert box.to_dict() == {
        "min_x": 0.0,
        "min_y": 1.0,
        "min_z": 2.0,
        "max_x": 10.0,
        "max_y": 5.0,
        "max_z": 8.0,
    }


def test_bbox3d_rejects_min_greater_than_max() -> None:
    with pytest.raises(ValueError, match="min_x"):
        BBox3D(10.0, 0.0, 0.0, 5.0, 5.0, 5.0)
    with pytest.raises(ValueError, match="min_y"):
        BBox3D(0.0, 10.0, 0.0, 5.0, 5.0, 5.0)
    with pytest.raises(ValueError, match="min_z"):
        BBox3D(0.0, 0.0, 10.0, 5.0, 5.0, 5.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_bbox3d_rejects_non_finite_coordinates(bad_value: float) -> None:
    with pytest.raises(TopoCoreError):
        BBox3D(bad_value, 0.0, 0.0, 10.0, 5.0, 2.0)


# ----------------------------------------------------------------------
# Cross-dimensional consistency.
# ----------------------------------------------------------------------


def test_bbox2d_and_bbox3d_agree_on_inclusive_boundary_convention() -> None:
    """Both classes independently implement <= bounds -- confirms they never silently diverge."""
    box2d = BBox2D(0.0, 0.0, 10.0, 10.0)
    box3d = BBox3D(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)

    assert box2d.contains(Point2D(10.0, 10.0)) is True
    assert box3d.contains(Point3D(10.0, 10.0, 10.0)) is True
