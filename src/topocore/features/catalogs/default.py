"""
topocore.features.catalogs.default
==================================

Default feature-code catalog.

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

DEFAULT_CODES: tuple[FeatureCodeDefinition, ...] = (
    FeatureCodeDefinition(
        code="CERCA",
        name="Fence",
        feature_type=FeatureType.FENCE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="CERCAS",
    ),
    FeatureCodeDefinition(
        code="ARBOL",
        name="Tree",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
)

__all__ = [
    "DEFAULT_CODES",
]
