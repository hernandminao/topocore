"""
topocore.features.catalogs.terrain
===================================

Bare terrain shot catalog.

These codes mark a point as a plain ground/terrain shot rather than a
feature to draw: "TN" (terreno natural), "TERRENO", "RASANTE", etc.
They exist in survey field code lists because a total station
operator still tags every shot with something, and it must be
possible to tell "this point is deliberately ground, no linework
needed" apart from "this point's code isn't in the registry" -- the
latter is a real gap (unrecognized code, unmatched), the former isn't
a gap at all.

``FeatureGeometryType.GROUND`` codes are excluded from
``feature_builder`` output but never touch the TIN pipeline in the
first place: the TIN is built directly from a ``SurveyPointSet``'s
X/Y/Z, regardless of code, ground or otherwise.

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

TERRAIN_CODES: tuple[FeatureCodeDefinition, ...] = (
    FeatureCodeDefinition(
        code="TN",
        name="Terreno natural",
        geometry_type=FeatureGeometryType.GROUND,
        layer="TERRENO",
        aliases=("TERRENO", "NATURAL"),
    ),
    FeatureCodeDefinition(
        code="RASANTE",
        name="Rasante / subrasante",
        geometry_type=FeatureGeometryType.GROUND,
        layer="TERRENO",
        aliases=("SUBRASANTE",),
    ),
    FeatureCodeDefinition(
        code="SUELO",
        name="Punto de suelo",
        geometry_type=FeatureGeometryType.GROUND,
        layer="TERRENO",
    ),
)

__all__ = [
    "TERRAIN_CODES",
]
