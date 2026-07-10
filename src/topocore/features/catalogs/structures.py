"""
topocore.features.catalogs.structures
=====================================

Structures feature catalog.

Contains common field codes used in topographic surveys of buildings,
civil structures, industrial facilities and urban infrastructure.

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

STRUCTURE_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Buildings
    #
    FeatureCodeDefinition(
        code="EDIF",
        name="Building",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="CASA",
        name="House",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="VIVIENDA",
        name="Residence",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="LOCAL",
        name="Commercial Building",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="BODEGA",
        name="Warehouse",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="GALPON",
        name="Warehouse",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="EDIFICACIONES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="NAVE",
        name="Industrial Building",
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
        geometry_type=FeatureGeometryType.LINE,
        layer="MUROS",
    ),
    FeatureCodeDefinition(
        code="MURCONT",
        name="Retaining Wall",
        geometry_type=FeatureGeometryType.LINE,
        layer="MUROS",
    ),
    FeatureCodeDefinition(
        code="TAPIA",
        name="Boundary Wall",
        geometry_type=FeatureGeometryType.LINE,
        layer="MUROS",
    ),
    FeatureCodeDefinition(
        code="CERRAMIENTO",
        name="Fence",
        geometry_type=FeatureGeometryType.LINE,
        layer="CERRAMIENTOS",
    ),
    FeatureCodeDefinition(
        code="REJA",
        name="Fence",
        geometry_type=FeatureGeometryType.LINE,
        layer="CERRAMIENTOS",
    ),
    FeatureCodeDefinition(
        code="MALLA",
        name="Wire Fence",
        geometry_type=FeatureGeometryType.LINE,
        layer="CERRAMIENTOS",
    ),
    #
    # Platforms
    #
    FeatureCodeDefinition(
        code="PLACA",
        name="Concrete Slab",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PLATAFORMAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PLATAFORMA",
        name="Platform",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PLATAFORMAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PATIO",
        name="Yard",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PLATAFORMAS",
        closed=True,
    ),
    #
    # Pools
    #
    FeatureCodeDefinition(
        code="PISCINA",
        name="Swimming Pool",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PISCINAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="TANQUE",
        name="Tank",
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
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ESTRUCTURAS",
    ),
    FeatureCodeDefinition(
        code="PILAR",
        name="Pillar",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ESTRUCTURAS",
    ),
    FeatureCodeDefinition(
        code="BASE",
        name="Foundation",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ESTRUCTURAS",
    ),
    FeatureCodeDefinition(
        code="ZAPATA",
        name="Footing",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ESTRUCTURAS",
    ),
    #
    # Vertical circulation
    #
    FeatureCodeDefinition(
        code="ESCALERA",
        name="Stair",
        geometry_type=FeatureGeometryType.LINE,
        layer="ESCALERAS",
    ),
    FeatureCodeDefinition(
        code="RAMPA",
        name="Ramp",
        geometry_type=FeatureGeometryType.LINE,
        layer="ESCALERAS",
    ),
    #
    # Doors and openings
    #
    FeatureCodeDefinition(
        code="PUERTA",
        name="Door",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="DETALLES",
    ),
    FeatureCodeDefinition(
        code="PORTON",
        name="Gate",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="DETALLES",
    ),
    FeatureCodeDefinition(
        code="VENTANA",
        name="Window",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="DETALLES",
    ),
    #
    # Roof
    #
    FeatureCodeDefinition(
        code="ALERO",
        name="Roof Edge",
        geometry_type=FeatureGeometryType.LINE,
        layer="CUBIERTAS",
    ),
    FeatureCodeDefinition(
        code="CUBIERTA",
        name="Roof",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="CUBIERTAS",
        closed=True,
    ),
)

__all__ = [
    "STRUCTURE_CODES",
]