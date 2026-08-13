"""
topocore.io.landxml.coordinates
=================================

LandXML coordinate text convention.

LandXML encodes point coordinates as whitespace-separated text
inside an element (e.g. ``<P id="1">775431.17 2603215.79 62.9</P>``),
always in **Northing Easting Elevation** order -- confirmed both by
the LandXML-1.2 schema documentation ("north east elev") and by
Autodesk's own Civil 3D LandXML support notes: "All coordinate
locations are always treated as Northing, Easting, Elevation (or
Y,X,Z)". This is the single most dangerous silent mistake when
ingesting LandXML: the text order is NOT "x y z".

This module is the only place that convention is applied, so it is
never repeated (and never risks diverging) between ``reader.py`` and
``writer.py``. Mapping used throughout:

    LandXML text:  "north east elev"  (or "north east" for 2D)
                          |
                          v
    Point3D:        x=east, y=north, z=elev

This keeps ``Point3D.x``/``SurveyPoint.x`` meaning Easting, and
``.y`` meaning Northing, consistent with the existing convention
already established in ``topocore.survey.models.SurveyPoint``.

A 2D point (two values, no elevation) is accepted and given
``z=0.0`` -- LandXML permits 2D points (e.g. plan-only control
points); this default is deliberate and documented, not a silent
failure, since ``Point3D``/``SurveyPoint`` have no optional-z
variant.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geometry.point3d import Point3D
from topocore.io.landxml.constants import DEFAULT_COORDINATE_PRECISION
from topocore.io.landxml.exceptions import LandXMLParseError

_MIN_VALUES = 2
_MAX_VALUES = 3


def parse_point_text(text: str) -> Point3D:
    """
    Parse LandXML coordinate text into a ``Point3D``.

    Parameters
    ----------
    text
        Element text content: two ("north east") or three
        ("north east elev") whitespace-separated numbers.

    Returns
    -------
    Point3D
        ``x=east``, ``y=north``, ``z=elev`` (``0.0`` if omitted).

    Raises
    ------
    LandXMLParseError
        If ``text`` does not contain two or three valid numbers.
        In particular, a point defined via a ``pntRef`` attribute
        instead of literal coordinate text (permitted by the LandXML
        schema, but not supported here -- see PR18B contract) has no
        text content and raises this with an explicit message rather
        than a generic "0 values" complaint.
    """
    parts = text.split()

    if not parts:
        raise LandXMLParseError(
            "Point has no coordinate text -- likely defined via a 'pntRef' "
            "attribute referencing another point, which this reader does "
            "not support (only literal 'north east [elev]' text)."
        )

    if not (_MIN_VALUES <= len(parts) <= _MAX_VALUES):
        raise LandXMLParseError(
            f"Expected 2 or 3 coordinate values in 'north east [elev]' order, got {len(parts)}: {text!r}"
        )

    try:
        north = float(parts[0])
        east = float(parts[1])
        elev = float(parts[2]) if len(parts) == _MAX_VALUES else 0.0
    except ValueError as exc:
        raise LandXMLParseError(f"Non-numeric coordinate value in {text!r}") from exc

    return Point3D(east, north, elev)


def format_point_text(
    point: Point3D,
    *,
    precision: int = DEFAULT_COORDINATE_PRECISION,
) -> str:
    """
    Format a ``Point3D`` as LandXML "north east elev" text.

    Parameters
    ----------
    point
        Point to format. ``point.x`` is written as Easting,
        ``point.y`` as Northing -- the inverse of
        ``parse_point_text``.
    precision
        Decimal places for each coordinate.

    Returns
    -------
    str
    """
    return f"{point.y:.{precision}f} {point.x:.{precision}f} {point.z:.{precision}f}"


__all__ = [
    "format_point_text",
    "parse_point_text",
]
