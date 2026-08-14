"""
topocore.io.landxml.codecs
============================

Small text codecs for LandXML attribute/element conventions other
than the "north east elev" point coordinate convention (see
``coordinates.py`` for that one).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.io.landxml.constants import DEFAULT_COORDINATE_PRECISION
from topocore.io.landxml.exceptions import LandXMLParseError


def parse_radius(text: str) -> float:
    """
    Parse a LandXML radius value, including the schema's ``"INF"``
    convention for an infinite (straight) radius.

    LandXML conforms to XML Schema's representation of infinity for
    numeric types, which allows both ``"INF"`` and ``"-INF"``. Only
    ``"INF"`` is accepted here: a negative radius has no meaning in
    our clothoid model (``SpiralElement`` requires a positive finite
    radius or ``math.inf``), so ``"-INF"`` is rejected explicitly
    rather than silently coerced to ``math.inf`` or ``-math.inf``.

    Raises
    ------
    LandXMLParseError
        If ``text`` is ``"-INF"``, or is not ``"INF"`` and not a
        valid number.
    """
    stripped = text.strip()

    if stripped == "INF":
        return math.inf

    if stripped == "-INF":
        raise LandXMLParseError(
            "Radius '-INF' is not supported: a negative radius has no meaning for "
            "SpiralElement, which only accepts a positive finite radius or 'INF'."
        )

    try:
        return float(stripped)
    except ValueError as exc:
        raise LandXMLParseError(f"Invalid radius value: {text!r}") from exc


def format_radius(value: float, *, precision: int = DEFAULT_COORDINATE_PRECISION) -> str:
    """
    Format a radius value, using LandXML's ``"INF"`` convention for
    ``math.inf``.
    """
    if math.isinf(value):
        if value < 0:
            raise LandXMLParseError("Cannot format a negative-infinite radius: LandXML has no representation for it.")
        return "INF"

    return f"{value:.{precision}f}"


def parse_rotation(text: str) -> bool:
    """
    Parse a LandXML ``rot`` attribute (``"cw"``/``"ccw"``) into
    TopoCore's ``clockwise: bool``.

    Raises
    ------
    LandXMLParseError
        If ``text`` is neither ``"cw"`` nor ``"ccw"``.
    """
    stripped = text.strip().lower()

    if stripped == "cw":
        return True

    if stripped == "ccw":
        return False

    raise LandXMLParseError(f"Invalid rotation value: {text!r} (expected 'cw' or 'ccw').")


def format_rotation(clockwise: bool) -> str:
    return "cw" if clockwise else "ccw"


def parse_station_elevation(text: str) -> tuple[float, float]:
    """
    Parse a LandXML ``<PVI>``/``<ParaCurve>`` text value: two
    whitespace-separated numbers, "station elevation" -- already in
    that literal order (unlike point coordinate text, there is no
    Northing/Easting swap to worry about here; station and elevation
    are unambiguous, unrelated axes).

    Raises
    ------
    LandXMLParseError
        If ``text`` does not contain exactly two valid numbers.
    """
    parts = text.split()

    if len(parts) != 2:
        raise LandXMLParseError(f"Expected 'station elevation' (2 values), got {len(parts)}: {text!r}")

    try:
        station = float(parts[0])
        elevation = float(parts[1])
    except ValueError as exc:
        raise LandXMLParseError(f"Non-numeric value in {text!r}") from exc

    return station, elevation


def format_station_elevation(
    station: float,
    elevation: float,
    *,
    precision: int = DEFAULT_COORDINATE_PRECISION,
) -> str:
    return f"{station:.{precision}f} {elevation:.{precision}f}"


__all__ = [
    "format_radius",
    "format_rotation",
    "format_station_elevation",
    "parse_radius",
    "parse_rotation",
    "parse_station_elevation",
]
