"""
topocore.features.catalogs.structures
=====================================

Structures feature catalog.

Note
----
MURO/TAPIA -> WALL and MURCONT -> RETAINING_WALL are LINE here
(surveyed wall axis), while PR15's WallDetector/RetainingWallDetector
produce POLYGON (convex hull of a vertical-facade cluster) -- both
representations are legitimate for the same physical object, hence
WALL/RETAINING_WALL accepting both GeometryTypes in
_EXPECTED_GEOMETRY.

CUBIERTA -> ROOF (POLYGON footprint) is the same object as PR15's
triangulated MESH roof, same precedent as WALL. ALERO -> ROOF_EDGE
(LINE) is the roof's *boundary*, not the roof itself -- kept as its
own type, same precedent as ROAD/PAVEMENT_EDGE.

`category` is BUILDING for every code here because that's what the
closed semantic matrix assigned, not because these codes happen to
live in structures.py -- category is a property of feature_type, not
of which catalog file declares the code.

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

STRUCTURE_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Buildings
    #
    FeatureCodeDefinition(
        code="EDIF",
        name="Building",
        feature_type=FeatureType.BUILDING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="CASA",
        name="House",
        feature_type=FeatureType.BUILDING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="VIVIENDA",
        name="Residence",
        feature_type=FeatureType.BUILDING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="LOCAL",
        name="Commercial Building",
        feature_type=FeatureType.BUILDING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="BODEGA",
        name="Warehouse",
        feature_type=FeatureType.BUILDING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="GALPON",
        name="Warehouse",
        feature_type=FeatureType.BUILDING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="NAVE",
        name="Industrial Building",
        feature_type=FeatureType.BUILDING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    #
    # Walls and fences
    #
    FeatureCodeDefinition(
        code="MURO",
        name="Wall",
        feature_type=FeatureType.WALL,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="MUROS",
    ),
    FeatureCodeDefinition(
        code="MURCONT",
        name="Retaining Wall",
        feature_type=FeatureType.RETAINING_WALL,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="MUROS",
    ),
    FeatureCodeDefinition(
        code="TAPIA",
        name="Boundary Wall",
        feature_type=FeatureType.WALL,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="MUROS",
    ),
    FeatureCodeDefinition(
        code="CERRAMIENTO",
        name="Fence",
        feature_type=FeatureType.FENCE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="CERRAMIENTOS",
    ),
    FeatureCodeDefinition(
        code="REJA",
        name="Fence",
        feature_type=FeatureType.FENCE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="CERRAMIENTOS",
    ),
    FeatureCodeDefinition(
        code="MALLA",
        name="Wire Fence",
        feature_type=FeatureType.FENCE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="CERRAMIENTOS",
    ),
    #
    # Platforms
    #
    FeatureCodeDefinition(
        code="PLACA",
        name="Concrete Slab",
        feature_type=FeatureType.HARDSCAPE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PLATAFORMAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PLATAFORMA",
        name="Platform",
        feature_type=FeatureType.HARDSCAPE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PLATAFORMAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PATIO",
        name="Yard",
        feature_type=FeatureType.HARDSCAPE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PLATAFORMAS",
        closed=True,
    ),
    #
    # Pools / tanks
    #
    FeatureCodeDefinition(
        code="PISCINA",
        name="Swimming Pool",
        feature_type=FeatureType.POOL,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PISCINAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="TANQUE",
        name="Tank",
        feature_type=FeatureType.TANK,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="TANQUES",
        closed=True,
    ),
    #
    # Structural elements
    #
    FeatureCodeDefinition(
        code="COLUMNA",
        name="Column",
        feature_type=FeatureType.STRUCTURAL_ELEMENT,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ESTRUCTURAS",
    ),
    FeatureCodeDefinition(
        code="PILAR",
        name="Pillar",
        feature_type=FeatureType.STRUCTURAL_ELEMENT,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ESTRUCTURAS",
    ),
    FeatureCodeDefinition(
        code="BASE",
        name="Foundation",
        feature_type=FeatureType.STRUCTURAL_ELEMENT,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ESTRUCTURAS",
    ),
    FeatureCodeDefinition(
        code="ZAPATA",
        name="Footing",
        feature_type=FeatureType.STRUCTURAL_ELEMENT,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ESTRUCTURAS",
    ),
    #
    # Vertical circulation
    #
    FeatureCodeDefinition(
        code="ESCALERA",
        name="Stair",
        feature_type=FeatureType.STAIR,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="ESCALERAS",
    ),
    FeatureCodeDefinition(
        code="RAMPA",
        name="Ramp",
        feature_type=FeatureType.RAMP,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="ESCALERAS",
    ),
    #
    # Doors, gates, and openings
    #
    FeatureCodeDefinition(
        code="PUERTA",
        name="Door",
        feature_type=FeatureType.OPENING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="DETALLES",
    ),
    FeatureCodeDefinition(
        code="PORTON",
        name="Gate",
        feature_type=FeatureType.GATE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CERRAMIENTOS",
    ),
    FeatureCodeDefinition(
        code="VENTANA",
        name="Window",
        feature_type=FeatureType.OPENING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="DETALLES",
    ),
    #
    # Roof
    #
    FeatureCodeDefinition(
        code="ALERO",
        name="Roof Edge",
        feature_type=FeatureType.ROOF_EDGE,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="CUBIERTAS",
    ),
    FeatureCodeDefinition(
        code="CUBIERTA",
        name="Roof",
        feature_type=FeatureType.ROOF,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="CUBIERTAS",
        closed=True,
    ),
)

__all__ = [
    "STRUCTURE_CODES",
]
