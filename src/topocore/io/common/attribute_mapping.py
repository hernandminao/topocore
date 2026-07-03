"""
topocore.io.common.attribute_mapping
===================================

Utilities for resolving source attribute names into canonical
PointAttribute values.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final

from topocore.pointcloud.attributes import PointAttribute


def _normalize(name: str) -> str:
    """
    Normalize an attribute name.
    """
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


_ATTRIBUTE_MAPPING: Final[
    dict[str, PointAttribute]
] = {

    # Coordinates

    "x": PointAttribute.X,
    "y": PointAttribute.Y,
    "z": PointAttribute.Z,

    "easting": PointAttribute.X,
    "northing": PointAttribute.Y,
    "height": PointAttribute.Z,
    "elevation": PointAttribute.Z,

    # Intensity

    "intensity": PointAttribute.INTENSITY,

    # Classification

    "classification": PointAttribute.CLASSIFICATION,

    # Returns

    "return_number": PointAttribute.RETURN_NUMBER,

    "number_of_returns":
        PointAttribute.NUMBER_OF_RETURNS,

    # GPS

    "gps_time": PointAttribute.GPS_TIME,

    "gpstime": PointAttribute.GPS_TIME,

    # User

    "user_data": PointAttribute.USER_DATA,

    # Source

    "point_source_id":
        PointAttribute.POINT_SOURCE_ID,

}


def resolve_attribute(
    name: str,
) -> PointAttribute | None:
    """
    Resolve a source attribute name.
    """
    return _ATTRIBUTE_MAPPING.get(
        _normalize(name)
    )


def has_attribute(
    name: str,
) -> bool:
    """
    Return whether an attribute is supported.
    """
    return resolve_attribute(name) is not None


__all__ = [
    "resolve_attribute",
    "has_attribute",
]