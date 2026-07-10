"""
Tests for topocore.geometry.bbox3d.
"""

from __future__ import annotations

import pytest

from topocore.geometry.bbox3d import BBox3D
from topocore.geometry.point3d import Point3D

# ==========================================================
# Construction
# ==========================================================


def test_create_bbox() -> None:
    bbox = BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        20.0,
        30.0,
    )

    assert bbox.min_x == 0.0
    assert bbox.min_y == 0.0
    assert bbox.min_z == 0.0

    assert bbox.max_x == 10.0
    assert bbox.max_y == 20.0
    assert bbox.max_z == 30.0


def test_invalid_x_coordinates() -> None:
    with pytest.raises(ValueError):
        BBox3D(
            10.0,
            0.0,
            0.0,
            5.0,
            10.0,
            10.0,
        )


def test_invalid_y_coordinates() -> None:
    with pytest.raises(ValueError):
        BBox3D(
            0.0,
            10.0,
            0.0,
            5.0,
            5.0,
            10.0,
        )


def test_invalid_z_coordinates() -> None:
    with pytest.raises(ValueError):
        BBox3D(
            0.0,
            0.0,
            10.0,
            5.0,
            5.0,
            5.0,
        )


# ==========================================================
# Properties
# ==========================================================


def test_width() -> None:
    bbox = BBox3D(
        2.0,
        3.0,
        4.0,
        12.0,
        13.0,
        14.0,
    )

    assert bbox.width == 10.0


def test_height() -> None:
    bbox = BBox3D(
        2.0,
        3.0,
        4.0,
        12.0,
        13.0,
        14.0,
    )

    assert bbox.height == 10.0


def test_depth() -> None:
    bbox = BBox3D(
        2.0,
        3.0,
        4.0,
        12.0,
        13.0,
        14.0,
    )

    assert bbox.depth == 10.0


def test_volume() -> None:
    bbox = BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        5.0,
        2.0,
    )

    assert bbox.volume == 100.0


def test_center() -> None:
    bbox = BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        20.0,
        30.0,
    )

    assert bbox.center == Point3D(
        5.0,
        10.0,
        15.0,
    )


def test_min_point() -> None:
    bbox = BBox3D(
        1.0,
        2.0,
        3.0,
        10.0,
        20.0,
        30.0,
    )

    assert bbox.min_point == Point3D(
        1.0,
        2.0,
        3.0,
    )


def test_max_point() -> None:
    bbox = BBox3D(
        1.0,
        2.0,
        3.0,
        10.0,
        20.0,
        30.0,
    )

    assert bbox.max_point == Point3D(
        10.0,
        20.0,
        30.0,
    )


# ==========================================================
# Spatial predicates
# ==========================================================


def test_contains_inside() -> None:
    bbox = BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
    )

    assert bbox.contains(
        Point3D(
            5.0,
            5.0,
            5.0,
        )
    )


def test_contains_border() -> None:
    bbox = BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
    )

    assert bbox.contains(
        Point3D(
            10.0,
            0.0,
            10.0,
        )
    )


def test_contains_outside() -> None:
    bbox = BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
    )

    assert not bbox.contains(
        Point3D(
            12.0,
            5.0,
            5.0,
        )
    )


def test_intersects_true() -> None:
    first = BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
    )

    second = BBox3D(
        5.0,
        5.0,
        5.0,
        15.0,
        15.0,
        15.0,
    )

    assert first.intersects(second)


def test_intersects_false() -> None:
    first = BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
    )

    second = BBox3D(
        11.0,
        11.0,
        11.0,
        20.0,
        20.0,
        20.0,
    )

    assert not first.intersects(second)


# ==========================================================
# Operations
# ==========================================================


def test_expand() -> None:
    bbox = BBox3D(
        2.0,
        2.0,
        2.0,
        8.0,
        8.0,
        8.0,
    )

    expanded = bbox.expand(
        2.0,
    )

    assert expanded == BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
    )


def test_expand_negative_margin() -> None:
    bbox = BBox3D(
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    )

    with pytest.raises(ValueError):
        bbox.expand(-1.0)


def test_union() -> None:
    first = BBox3D(
        0.0,
        0.0,
        0.0,
        5.0,
        5.0,
        5.0,
    )

    second = BBox3D(
        4.0,
        3.0,
        2.0,
        10.0,
        12.0,
        14.0,
    )

    result = first.union(second)

    assert result == BBox3D(
        0.0,
        0.0,
        0.0,
        10.0,
        12.0,
        14.0,
    )


# ==========================================================
# Serialization
# ==========================================================


def test_to_dict() -> None:
    bbox = BBox3D(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    )

    assert bbox.to_dict() == {
        "min_x": 1.0,
        "min_y": 2.0,
        "min_z": 3.0,
        "max_x": 4.0,
        "max_y": 5.0,
        "max_z": 6.0,
    }
