"""
Regression suite for topocore.io.landxml.coordinates -- PR19.

Verifies the single most dangerous convention in this module (per
its own docstring): LandXML text order is "north east elev", NOT
"x y z". Confirmed with a real-world-style coordinate string that
parse_point_text/format_point_text correctly map x=east, y=north,
z=elev, with an exact numeric round trip. No bugs found (re-confirms
PR18B's own work).
"""

from __future__ import annotations

import pytest

from topocore.io.landxml.coordinates import format_point_text, parse_point_text
from topocore.io.landxml.exceptions import LandXMLParseError


def test_north_east_elev_order_mapped_correctly() -> None:
    point = parse_point_text("775431.17 2603215.79 62.9")
    assert point.x == pytest.approx(2603215.79)  # easting (2nd value)
    assert point.y == pytest.approx(775431.17)  # northing (1st value)
    assert point.z == pytest.approx(62.9)


def test_round_trip_numeric_value_preserved() -> None:
    point = parse_point_text("775431.17 2603215.79 62.9")
    formatted = format_point_text(point, precision=2)
    assert formatted == "775431.17 2603215.79 62.90"


def test_2d_point_defaults_elevation_to_zero() -> None:
    point = parse_point_text("775431.17 2603215.79")
    assert point.z == 0.0


def test_pntref_style_empty_text_raises_clear_error() -> None:
    with pytest.raises(LandXMLParseError):
        parse_point_text("")


def test_rejects_non_numeric_values() -> None:
    with pytest.raises(LandXMLParseError):
        parse_point_text("abc def")


def test_rejects_wrong_value_count() -> None:
    with pytest.raises(LandXMLParseError):
        parse_point_text("1.0 2.0 3.0 4.0")
