"""
topocore.features
==================

Field-to-finish geometry construction: turns coded survey points into
CAD/GIS-ready features (lines, polygons, symbols).

Implemented so far
-------------------

* ``feature_codes``   -- the code -> geometry-type dictionary
* ``feature_builder`` -- groups survey points by code into geometry

Planned, not yet implemented
------------------------------

* ``line_builder`` / ``polygon_builder`` -- geometry refinement
  (curve fitting, offset lines, closing tolerances)
* ``symbol_builder``  -- CAD block/symbol placement
* ``cad_layers`` / ``gis_layers`` -- layer/table mapping for DXF
  (PR16) and GeoPackage (PR17) export
* ``feature_classifier`` -- automatic classification for uncoded /
  LiDAR-derived points, complementing this code-driven path

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .feature_builder import FeatureBuilder, FeatureSet, LineFeature, PointFeature, build_features
from .feature_codes import FeatureCodeDefinition, FeatureCodeRegistry, FeatureGeometryType

__all__ = [
    "FeatureGeometryType",
    "FeatureCodeDefinition",
    "FeatureCodeRegistry",
    "PointFeature",
    "LineFeature",
    "FeatureSet",
    "FeatureBuilder",
    "build_features",
]
