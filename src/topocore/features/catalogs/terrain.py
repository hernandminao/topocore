"""
topocore.features.catalogs.terrain
====================================

Bare terrain-shot codes. All three use ``geometry_type=GROUND`` and
``feature_type=None`` -- they feed TIN/DTM construction directly and
never produce a `Feature`; see the invariant enforced by
``catalogs._validation.validate_definition_geometry``.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.feature_codes import FeatureCodeDefinition, FeatureGeometryType
from topocore.features.models import FeatureCategory

TERRAIN_CODES: tuple[FeatureCodeDefinition, ...] = (
    FeatureCodeDefinition(
        code="TN",
        name="Terreno natural",
        feature_type=None,
        category=FeatureCategory.TERRAIN,
        geometry_type=FeatureGeometryType.GROUND,
        layer="TERRENO",
    ),
    FeatureCodeDefinition(
        code="RASANTE",
        name="Rasante / subrasante",
        feature_type=None,
        category=FeatureCategory.TERRAIN,
        geometry_type=FeatureGeometryType.GROUND,
        layer="TERRENO",
    ),
    FeatureCodeDefinition(
        code="SUELO",
        name="Punto de suelo",
        feature_type=None,
        category=FeatureCategory.TERRAIN,
        geometry_type=FeatureGeometryType.GROUND,
        layer="TERRENO",
    ),
)

__all__ = [
    "TERRAIN_CODES",
]
