"""
topocore.features.catalogs.control
==================================

Survey control catalog.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.feature_codes import (
    FeatureCodeDefinition,
    FeatureGeometryType,
)

CONTROL_CODES: tuple[FeatureCodeDefinition, ...] = (
    FeatureCodeDefinition(
        code="BM",
        name="Benchmark",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="VERTICE",
        name="Geodetic Vertex",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="MOJON",
        name="Boundary Monument",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="GPS",
        name="Point Control",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
)

__all__ = [
    "CONTROL_CODES",
]
