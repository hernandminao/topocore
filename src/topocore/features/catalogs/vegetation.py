"""
topocore.features.catalogs.vegetation
=======================================

Vegetation feature catalog.

Note
----
ROCAARBOL is intentionally NOT migrated yet -- its declared name
alone ("Tree on Rock") doesn't disambiguate what it marks (the rock,
the tree, or a composite feature), and inventing a FeatureType from
the name alone risks getting the semantics wrong. It remains absent
from VEGETATION_CODES until its real-world meaning is confirmed.

TRONCO, TOCON, and RAIZ were previously deferred alongside
ROCAARBOL, but each has an unambiguous, specific meaning (trunk,
stump, root), so they were promoted to their own FeatureType
(TREE_TRUNK, TREE_STUMP, TREE_ROOT) rather than left pending or
folded into TREE -- collapsing them would have lost real field
semantics TopoCore is meant to preserve.

ARBOL is intentionally duplicated here with an identical definition
to catalogs.default.ARBOL (same name/geometry_type/layer, same
feature_type/category) -- FeatureCodeRegistry.register() accepts
re-registering an identical definition, so this is harmless, though
redundant at the source level; a future cleanup could make
default.py the single source of truth and have vegetation.py stop
redeclaring it.

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

VEGETATION_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Trees
    #
    FeatureCodeDefinition(
        code="ARBOL",
        name="Tree",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="ARBOLSEC",
        name="Secondary Tree",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="ARBOLAIS",
        name="Isolated Tree",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="PALMA",
        name="Palm Tree",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="PINO",
        name="Pine Tree",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="EUCALIPTO",
        name="Eucalyptus",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="GUADUA",
        name="Guadua",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    #
    # Shrubs
    #
    FeatureCodeDefinition(
        code="ARBUSTO",
        name="Shrub",
        feature_type=FeatureType.SHRUB,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="MATORRAL",
        name="Brush",
        feature_type=FeatureType.SHRUB,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    #
    # Vegetation lines
    #
    FeatureCodeDefinition(
        code="CERCAVIVA",
        name="Live Fence",
        feature_type=FeatureType.VEGETATION_LINE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.LINE,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="SETO",
        name="Hedge",
        feature_type=FeatureType.VEGETATION_LINE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.LINE,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="LINEAARBOL",
        name="Tree Line",
        feature_type=FeatureType.VEGETATION_LINE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.LINE,
        layer="VEGETACION",
    ),
    #
    # Forest / crop areas
    #
    FeatureCodeDefinition(
        code="BOSQUE",
        name="Forest",
        feature_type=FeatureType.FOREST,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="SELVA",
        name="Jungle",
        feature_type=FeatureType.FOREST,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="CULTIVO",
        name="Crop Area",
        feature_type=FeatureType.CROP,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="CULTIVOS",
        closed=True,
    ),
    #
    # Grass / groundcover areas
    #
    FeatureCodeDefinition(
        code="PASTO",
        name="Grass Area",
        feature_type=FeatureType.GRASS,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PRADO",
        name="Lawn",
        feature_type=FeatureType.GRASS,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="JARDIN",
        name="Garden",
        feature_type=FeatureType.GRASS,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="ZONAVERDE",
        name="Green Area",
        feature_type=FeatureType.GRASS,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="VEGETACION",
        closed=True,
    ),
    #
    # Individual tree parts
    #
    FeatureCodeDefinition(
        code="TRONCO",
        name="Tree Trunk",
        feature_type=FeatureType.TREE_TRUNK,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="TOCON",
        name="Tree Stump",
        feature_type=FeatureType.TREE_STUMP,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    FeatureCodeDefinition(
        code="RAIZ",
        name="Root",
        feature_type=FeatureType.TREE_ROOT,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="VEGETACION",
    ),
    # ROCAARBOL ("Tree on Rock"): deferred, see module docstring.
)

__all__ = [
    "VEGETATION_CODES",
]
