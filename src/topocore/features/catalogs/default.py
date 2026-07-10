"""
topocore.features.catalogs.default
==================================

Default feature-code catalog.

This module contains the minimal feature definitions shipped with
TopoCore. These definitions represent the most common field codes used
in general topographic surveys.

Additional catalogs (roads, utilities, vegetation, etc.) extend this
base catalog.

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

DEFAULT_CODES: tuple[FeatureCodeDefinition, ...] = (
    FeatureCodeDefinition(
        code="CERCA",
        name="Fence",
        geometry_type=FeatureGeometryType.LINE,
        layer="CERCAS",
    ),
    FeatureCodeDefinition(
        code="MURO",
        name="Wall",
        geometry_type=FeatureGeometryType.LINE,
        layer="MUROS",
    ),
    FeatureCodeDefinition(
        code="ARBOL",
        name="Tree",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="POSTE",
        name="Utility Pole",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="SERVICIOS",
    ),
)

__all__ = [
    "DEFAULT_CODES",
]