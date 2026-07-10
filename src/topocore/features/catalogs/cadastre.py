"""
topocore.features.catalogs.cadastre
===================================

Cadastre and property boundary feature catalog.

Contains common field codes used in cadastral, property, legal,
boundary and land surveying.

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

CADASTRE_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Property boundaries
    #
    FeatureCodeDefinition(
        code="LOTE",
        name="Property",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PREDIOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PREDIO",
        name="Property",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PREDIOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PARCELA",
        name="Parcel",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PREDIOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="FINCA",
        name="Farm",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PREDIOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="MANZANA",
        name="Block",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="MANZANAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="SECTOR",
        name="Sector",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="SECTORES",
        closed=True,
    ),
    #
    # Boundary lines
    #
    FeatureCodeDefinition(
        code="LINDERO",
        name="Boundary",
        geometry_type=FeatureGeometryType.LINE,
        layer="LINDEROS",
    ),
    FeatureCodeDefinition(
        code="LIMITE",
        name="Boundary",
        geometry_type=FeatureGeometryType.LINE,
        layer="LINDEROS",
    ),
    FeatureCodeDefinition(
        code="DESLINDE",
        name="Boundary",
        geometry_type=FeatureGeometryType.LINE,
        layer="LINDEROS",
    ),
    FeatureCodeDefinition(
        code="SERVIDUMBRE",
        name="Easement",
        geometry_type=FeatureGeometryType.LINE,
        layer="SERVIDUMBRES",
    ),
    FeatureCodeDefinition(
        code="RETIRO",
        name="Setback",
        geometry_type=FeatureGeometryType.LINE,
        layer="RETIROS",
    ),
    #
    # Boundary monuments
    #
    FeatureCodeDefinition(
        code="MOJON",
        name="Boundary Monument",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="HITO",
        name="Boundary Marker",
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
        code="BM",
        name="Benchmark",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="PI",
        name="Intersection Point",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    #
    # Administrative
    #
    FeatureCodeDefinition(
        code="MUNICIPIO",
        name="Municipality Boundary",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ADMINISTRATIVO",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="VEREDA",
        name="Village",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ADMINISTRATIVO",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="BARRIO",
        name="Neighborhood",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ADMINISTRATIVO",
        closed=True,
    ),
    #
    # Rights of way
    #
    FeatureCodeDefinition(
        code="FAJA",
        name="Right of Way",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="SERVIDUMBRES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="ZONA",
        name="Zone",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ZONAS",
        closed=True,
    ),
    #
    # Reference points
    #
    FeatureCodeDefinition(
        code="ESQUINA",
        name="Corner",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="REFERENCIAS",
    ),
    FeatureCodeDefinition(
        code="REFERENCIA",
        name="Reference Point",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="REFERENCIAS",
    ),
    FeatureCodeDefinition(
        code="CLAVO",
        name="Survey Nail",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="REFERENCIAS",
    ),
    FeatureCodeDefinition(
        code="ESTACA",
        name="Stake",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="REFERENCIAS",
    ),
    #
    # Miscellaneous
    #
    FeatureCodeDefinition(
        code="PORTON",
        name="Gate",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CERRAMIENTOS",
    ),
    FeatureCodeDefinition(
        code="CERCA",
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
)

__all__ = [
    "CADASTRE_CODES",
]