"""
topocore.features
====================

Automatic geospatial feature extraction from point clouds, TINs,
and DTMs — the bridge between raw processed point clouds and
CAD/GIS deliverables.

Every detector returns a `FeatureCollection` (see `models.py`),
providing a common representation for downstream consumers such as
the DXF and GeoPackage exporters planned for PR16 and PR17.

`FeatureGeometry` provides the geometric representation required by
those exporters, while `Feature.attributes` and `FeatureMetadata`
carry semantic attributes and detector provenance.

Importing this package registers every built-in detector with
`DetectorRegistry` — `FeatureExtractionManager.available_detectors`
is populated as a side effect of importing the detector
subpackages below.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from . import buildings, drainage, infrastructure, terrain, utilities, vegetation
from .base import BaseFeatureDetector
from .detector import DetectorRegistry
from .exceptions import DetectionError, FeatureError, GeometryError
from .manager import FeatureExtractionManager
from .models import (
    ContextField,
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureMetadata,
    FeatureType,
    GeometryType,
)
from .protocols import DetectionContext, FeatureDetectorProtocol

__all__ = [
    "FeatureExtractionManager",
    "DetectionContext",
    "FeatureDetectorProtocol",
    "BaseFeatureDetector",
    "DetectorRegistry",
    "Feature",
    "FeatureCollection",
    "FeatureGeometry",
    "FeatureMetadata",
    "FeatureType",
    "FeatureCategory",
    "GeometryType",
    "ContextField",
    "FeatureError",
    "GeometryError",
    "DetectionError",
    # Subpackages imported for detector-registration side effects.
    "terrain",
    "buildings",
    "infrastructure",
    "drainage",
    "vegetation",
    "utilities",
]
