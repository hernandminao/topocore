"""
Geodesy package for TopoCore.

Provides an abstraction layer over pyproj for Coordinate Reference Systems,
transformations, and geodesic calculations.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .crs import CRS
from .datum import Datum
from .distance import distance2d, distance3d
from .ellipsoid import Ellipsoid
from .exceptions import (
    CRSError,
    GeodesicError,
    GeodesyError,
    TransformationError,
    ValidationError,
)
from .geodesic import GeodesicCalculator
from .projection import ProjectionInfo
from .transformer import CoordinateTransformer
from .utm import UTMZone
from .validation import (
    validate_array,
    validate_bbox,
    validate_coordinate_arrays,
    validate_epsg,
    validate_lat_lon,
)

__all__ = [
    "CRS",
    "CoordinateTransformer",
    "Datum",
    "Ellipsoid",
    "GeodesicCalculator",
    "GeodesyError",
    "CRSError",
    "TransformationError",
    "GeodesicError",
    "ValidationError",
    "validate_coordinate_arrays",
    "ProjectionInfo",
    "UTMZone",
    "distance2d",
    "distance3d",
    "validate_lat_lon",
    "validate_epsg",
    "validate_bbox",
    "validate_array",
]
