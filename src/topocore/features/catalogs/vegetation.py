"""
topocore.features.catalogs.vegetation
=====================================

Vegetation feature catalog.

Contains common field codes used in forestry, environmental,
landscape and topographic surveys.

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

VEGETATION_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Trees
    #
    FeatureCodeDefinition(
        code="ARBOL",
        name="Tree",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="ARBOLSEC",
        name="Secondary Tree",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="ARBOLAIS",
        name="Isolated Tree",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="PALMA",
        name="Palm Tree",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="PINO",
        name="Pine Tree",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="EUCALIPTO",
        name="Eucalyptus",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="GUADUA",
        name="Guadua",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    #
    # Shrubs
    #
    FeatureCodeDefinition(
        code="ARBUSTO",
        name="Shrub",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="MATORRAL",
        name="Brush",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    #
    # Vegetation lines
    #
    FeatureCodeDefinition(
        code="CERCAVIVA",
        name="Live Fence",
        geometry_type=FeatureGeometryType.LINE,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="SETO",
        name="Hedge",
        geometry_type=FeatureGeometryType.LINE,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="LINEAARBOL",
        name="Tree Line",
        geometry_type=FeatureGeometryType.LINE,
        layer="VEGETACION",
    ),
    #
    # Vegetation areas
    #
    FeatureCodeDefinition(
        code="BOSQUE",
        name="Forest",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="SELVA",
        name="Jungle",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="CULTIVO",
        name="Crop Area",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="CULTIVOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PASTO",
        name="Grass Area",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PRADO",
        name="Lawn",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="JARDIN",
        name="Garden",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="ZONAVERDE",
        name="Green Area",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    #
    # Individual vegetation features
    #
    FeatureCodeDefinition(
        code="TRONCO",
        name="Tree Trunk",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="TOCON",
        name="Tree Stump",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="RAIZ",
        name="Root",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="ROCAARBOL",
        name="Tree on Rock",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
)

__all__ = [
    "VEGETATION_CODES",
]