"""
topocore.features.catalogs.drainage
===================================

Drainage and hydrography feature catalog.

Contains common field codes used in drainage, sewer, stormwater,
hydrology and hydraulic surveys.

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

DRAINAGE_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Natural drainage
    #
    FeatureCodeDefinition(
        code="RIO",
        name="River",
        geometry_type=FeatureGeometryType.LINE,
        layer="HIDROGRAFIA",
    ),
    FeatureCodeDefinition(
        code="QUEBRADA",
        name="Stream",
        geometry_type=FeatureGeometryType.LINE,
        layer="HIDROGRAFIA",
    ),
    FeatureCodeDefinition(
        code="CAÑO",
        name="Creek",
        geometry_type=FeatureGeometryType.LINE,
        layer="HIDROGRAFIA",
    ),
    FeatureCodeDefinition(
        code="ARROYO",
        name="Creek",
        geometry_type=FeatureGeometryType.LINE,
        layer="HIDROGRAFIA",
    ),
    FeatureCodeDefinition(
        code="LAGUNA",
        name="Lagoon",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="HIDROGRAFIA",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="LAGO",
        name="Lake",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="HIDROGRAFIA",
        closed=True,
    ),
    #
    # Artificial drainage
    #
    FeatureCodeDefinition(
        code="CANAL",
        name="Channel",
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    FeatureCodeDefinition(
        code="ACEQUIA",
        name="Irrigation Canal",
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    FeatureCodeDefinition(
        code="ZANJA",
        name="Ditch",
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    FeatureCodeDefinition(
        code="CUNETA",
        name="Road Ditch",
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    #
    # Sewer network
    #
    FeatureCodeDefinition(
        code="ALCANTARILLA",
        name="Culvert",
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="BOX",
        name="Box Culvert",
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="TUBERIA",
        name="Drain Pipe",
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="COLECTOR",
        name="Collector Sewer",
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    #
    # Inspection structures
    #
    FeatureCodeDefinition(
        code="POZO",
        name="Inspection Manhole",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="MH",
        name="Manhole",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="SUMIDERO",
        name="Storm Drain Inlet",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="REJILLA",
        name="Drain Grate",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="CAMARA",
        name="Inspection Chamber",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    #
    # Hydraulic structures
    #
    FeatureCodeDefinition(
        code="VERTEDERO",
        name="Spillway",
        geometry_type=FeatureGeometryType.LINE,
        layer="OBRAS_HIDRAULICAS",
    ),
    FeatureCodeDefinition(
        code="DIQUE",
        name="Levee",
        geometry_type=FeatureGeometryType.LINE,
        layer="OBRAS_HIDRAULICAS",
    ),
    FeatureCodeDefinition(
        code="REPRESA",
        name="Dam",
        geometry_type=FeatureGeometryType.LINE,
        layer="OBRAS_HIDRAULICAS",
    ),
    FeatureCodeDefinition(
        code="EMBALSE",
        name="Reservoir",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="OBRAS_HIDRAULICAS",
        closed=True,
    ),
)

__all__ = [
    "DRAINAGE_CODES",
]