"""
Definitions of generic point cloud attributes.

This module defines the canonical attributes supported by TopoCore,
independent of any particular point cloud file format.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any
from typing import Final

import numpy as np


class AttributeRequirement(Enum):
    """
    Defines whether an attribute is mandatory.
    """

    REQUIRED = auto()

    OPTIONAL = auto()

    FORMAT_DEPENDENT = auto()


class PointAttribute(Enum):
    """
    Generic point cloud attributes.

    These identifiers are independent of LAS, LAZ, E57, PLY,
    XYZ or any other storage format.
    """

    X = "x"
    Y = "y"
    Z = "z"

    INTENSITY = "intensity"

    RETURN_NUMBER = "return_number"

    NUMBER_OF_RETURNS = "number_of_returns"

    CLASSIFICATION = "classification"

    SCAN_ANGLE = "scan_angle"

    GPS_TIME = "gps_time"

    COLOR = "color"

    NIR = "nir"

    NORMAL = "normal"

    USER_DATA = "user_data"

    POINT_SOURCE_ID = "point_source_id"


@dataclass(frozen=True, slots=True)
class AttributeInfo:
    """
    Metadata describing a point cloud attribute.
    """

    attribute: PointAttribute

    dtype: np.dtype[Any]

    shape: tuple[int, ...]

    requirement: AttributeRequirement

    description: str


ATTRIBUTE_DEFINITIONS: Final[
    dict[PointAttribute, AttributeInfo]
] = {

    PointAttribute.X: AttributeInfo(
        attribute=PointAttribute.X,
        dtype=np.dtype(np.float64),
        shape=(1,),
        requirement=AttributeRequirement.REQUIRED,
        description="X coordinate.",
    ),

    PointAttribute.Y: AttributeInfo(
        attribute=PointAttribute.Y,
        dtype=np.dtype(np.float64),
        shape=(1,),
        requirement=AttributeRequirement.REQUIRED,
        description="Y coordinate.",
    ),

    PointAttribute.Z: AttributeInfo(
        attribute=PointAttribute.Z,
        dtype=np.dtype(np.float64),
        shape=(1,),
        requirement=AttributeRequirement.REQUIRED,
        description="Z coordinate.",
    ),

    PointAttribute.INTENSITY: AttributeInfo(
        attribute=PointAttribute.INTENSITY,
        dtype=np.dtype(np.uint16),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="Pulse intensity.",
    ),

    PointAttribute.RETURN_NUMBER: AttributeInfo(
        attribute=PointAttribute.RETURN_NUMBER,
        dtype=np.dtype(np.uint8),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="Return number.",
    ),

    PointAttribute.NUMBER_OF_RETURNS: AttributeInfo(
        attribute=PointAttribute.NUMBER_OF_RETURNS,
        dtype=np.dtype(np.uint8),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="Number of returns.",
    ),

    PointAttribute.CLASSIFICATION: AttributeInfo(
        attribute=PointAttribute.CLASSIFICATION,
        dtype=np.dtype(np.uint8),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="Point classification.",
    ),

    PointAttribute.SCAN_ANGLE: AttributeInfo(
        attribute=PointAttribute.SCAN_ANGLE,
        dtype=np.dtype(np.int16),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="Scan angle.",
    ),

    PointAttribute.GPS_TIME: AttributeInfo(
        attribute=PointAttribute.GPS_TIME,
        dtype=np.dtype(np.float64),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="GPS time.",
    ),

    PointAttribute.COLOR: AttributeInfo(
        attribute=PointAttribute.COLOR,
        dtype=np.dtype(np.uint16),
        shape=(3,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="RGB color.",
    ),

    PointAttribute.NIR: AttributeInfo(
        attribute=PointAttribute.NIR,
        dtype=np.dtype(np.uint16),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="Near infrared value.",
    ),

    PointAttribute.NORMAL: AttributeInfo(
        attribute=PointAttribute.NORMAL,
        dtype=np.dtype(np.float32),
        shape=(3,),
        requirement=AttributeRequirement.OPTIONAL,
        description="Surface normal vector.",
    ),

    PointAttribute.USER_DATA: AttributeInfo(
        attribute=PointAttribute.USER_DATA,
        dtype=np.dtype(np.uint8),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="User-defined value.",
    ),

    PointAttribute.POINT_SOURCE_ID: AttributeInfo(
        attribute=PointAttribute.POINT_SOURCE_ID,
        dtype=np.dtype(np.uint16),
        shape=(1,),
        requirement=AttributeRequirement.FORMAT_DEPENDENT,
        description="Point source identifier.",
    ),
}


def get_attribute_info(
    attribute: PointAttribute,
) -> AttributeInfo:
    """
    Returns the metadata associated with an attribute.
    """
    return ATTRIBUTE_DEFINITIONS[attribute]


ATTRIBUTE_DTYPES: Final[
    dict[PointAttribute, np.dtype[Any]]
] = {
    attribute: info.dtype
    for attribute, info in ATTRIBUTE_DEFINITIONS.items()
}


__all__ = [
    "AttributeInfo",
    "AttributeRequirement",
    "PointAttribute",
    "ATTRIBUTE_DEFINITIONS",
    "ATTRIBUTE_DTYPES",
    "get_attribute_info",
]