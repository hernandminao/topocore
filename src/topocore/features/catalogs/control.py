"""
topocore.features.catalogs.control
====================================

Geodetic/survey control point catalog.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.feature_codes import FeatureCodeDefinition, FeatureGeometryType
from topocore.features.models import FeatureCategory, FeatureType

CONTROL_CODES: tuple[FeatureCodeDefinition, ...] = (
    FeatureCodeDefinition(
        code="BM",
        name="Benchmark",
        feature_type=FeatureType.CONTROL_POINT,
        category=FeatureCategory.CONTROL,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="VERTICE",
        name="Geodetic Vertex",
        feature_type=FeatureType.CONTROL_POINT,
        category=FeatureCategory.CONTROL,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="MOJON",
        name="Boundary Monument",
        feature_type=FeatureType.BOUNDARY_MONUMENT,
        category=FeatureCategory.CONTROL,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="GPS",
        name="Point Control",
        feature_type=FeatureType.CONTROL_POINT,
        category=FeatureCategory.CONTROL,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
)

__all__ = [
    "CONTROL_CODES",
]
