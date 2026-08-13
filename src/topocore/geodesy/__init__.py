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

from .accuracy import TransformationAccuracy
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
from .grid_shift import GridShift, GridShiftFormat
from .helmert import HelmertParameters
from .local_crs import LocalCRS
from .operation import CoordinateOperation
from .operation_type import OperationType
from .projection import ProjectionInfo
from .transform import (
    transform_feature_collection,
    transform_point_cloud,
    transform_survey,
)
from .transformer import CoordinateTransformer
from .utm import UTMZone
from .validation import (
    validate_array,
    validate_bbox,
    validate_coordinate_arrays,
    validate_epsg,
    validate_lat_lon,
)
from .vertical_datum import VerticalDatum

__all__ = [
    "CRS",
    "CRSError",
    "CoordinateOperation",
    "CoordinateTransformer",
    "Datum",
    "Ellipsoid",
    "GeodesicCalculator",
    "GeodesicError",
    "GeodesyError",
    "GridShift",
    "GridShiftFormat",
    "HelmertParameters",
    "LocalCRS",
    "OperationType",
    "ProjectionInfo",
    "TransformationAccuracy",
    "TransformationError",
    "UTMZone",
    "ValidationError",
    "VerticalDatum",
    "distance2d",
    "distance3d",
    "transform_feature_collection",
    "transform_point_cloud",
    "transform_survey",
    "validate_array",
    "validate_bbox",
    "validate_coordinate_arrays",
    "validate_epsg",
    "validate_lat_lon",
]
