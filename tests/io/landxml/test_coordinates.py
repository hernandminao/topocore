"""
Tests for topocore.io.landxml.coordinates.

The single most important thing this module gets right: LandXML
coordinate text is "north east elev", not "x y z".
"""

from __future__ import annotations

import pytest

from topocore.io.landxml.coordinates import format_point_text, parse_point_text
from topocore.io.landxml.exceptions import LandXMLParseError


def test_parse_point_text_maps_north_east_elev_to_xyz() -> None:
    # north=10 (-> y), east=20 (-> x), elev=30 (-> z)
    point = parse_point_text("10.0 20.0 30.0")

    assert point.x == pytest.approx(20.0)
    assert point.y == pytest.approx(10.0)
    assert point.z == pytest.approx(30.0)


def test_parse_point_text_2d_defaults_z_to_zero() -> None:
    point = parse_point_text("10.0 20.0")

    assert point.x == pytest.approx(20.0)
    assert point.y == pytest.approx(10.0)
    assert point.z == pytest.approx(0.0)


def test_parse_point_text_rejects_one_value() -> None:
    with pytest.raises(LandXMLParseError):
        parse_point_text("10.0")


def test_parse_point_text_rejects_four_values() -> None:
    with pytest.raises(LandXMLParseError):
        parse_point_text("10.0 20.0 30.0 40.0")


def test_parse_point_text_rejects_non_numeric() -> None:
    with pytest.raises(LandXMLParseError):
        parse_point_text("abc 20.0 30.0")


def test_format_point_text_is_inverse_of_parse() -> None:
    original = "775431.17036070 2603215.78860469 62.90000000"

    point = parse_point_text(original)
    formatted = format_point_text(point)

    round_tripped = parse_point_text(formatted)

    assert round_tripped.x == pytest.approx(point.x)
    assert round_tripped.y == pytest.approx(point.y)
    assert round_tripped.z == pytest.approx(point.z)


def test_format_point_text_writes_north_east_order() -> None:
    from topocore.geometry.point3d import Point3D

    point = Point3D(x=20.0, y=10.0, z=30.0)  # east=20, north=10, elev=30

    text = format_point_text(point, precision=1)

    assert text == "10.0 20.0 30.0"
