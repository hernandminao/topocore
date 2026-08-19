"""
Tests for topocore.io.landxml.codecs.
"""

from __future__ import annotations

import math

import pytest

from topocore.io.landxml.codecs import (
    format_radius,
    format_rotation,
    format_station_elevation,
    parse_radius,
    parse_rotation,
    parse_station_elevation,
)
from topocore.io.landxml.exceptions import LandXMLParseError


def test_parse_radius_inf() -> None:
    assert parse_radius("INF") == math.inf


def test_parse_radius_rejects_negative_inf() -> None:
    with pytest.raises(LandXMLParseError, match="-INF"):
        parse_radius("-INF")


def test_parse_radius_finite() -> None:
    assert parse_radius("150.5") == pytest.approx(150.5)


def test_parse_radius_rejects_non_numeric() -> None:
    with pytest.raises(LandXMLParseError):
        parse_radius("abc")


def test_format_radius_inf() -> None:
    assert format_radius(math.inf) == "INF"


def test_format_radius_finite() -> None:
    assert format_radius(150.0, precision=2) == "150.00"


def test_format_radius_rejects_negative_inf() -> None:
    with pytest.raises(LandXMLParseError):
        format_radius(-math.inf)


def test_parse_rotation_cw() -> None:
    assert parse_rotation("cw") is True


def test_parse_rotation_ccw() -> None:
    assert parse_rotation("ccw") is False


def test_parse_rotation_rejects_invalid() -> None:
    with pytest.raises(LandXMLParseError):
        parse_rotation("clockwise")


def test_format_rotation_roundtrips() -> None:
    assert parse_rotation(format_rotation(True)) is True
    assert parse_rotation(format_rotation(False)) is False


def test_parse_station_elevation() -> None:
    station, elevation = parse_station_elevation("1250.0 102.4")
    assert station == pytest.approx(1250.0)
    assert elevation == pytest.approx(102.4)


def test_parse_station_elevation_rejects_wrong_count() -> None:
    with pytest.raises(LandXMLParseError):
        parse_station_elevation("1250.0 102.4 5.0")


def test_format_station_elevation_roundtrips() -> None:
    text = format_station_elevation(1250.0, 102.4, precision=2)
    station, elevation = parse_station_elevation(text)
    assert station == pytest.approx(1250.0)
    assert elevation == pytest.approx(102.4)
