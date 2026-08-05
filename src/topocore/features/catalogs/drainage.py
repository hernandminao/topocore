"""
topocore.features.catalogs.drainage
=====================================

Drainage and hydrography feature catalog.

Note
----
RIO/QUEBRADA/CAÑO/ARROYO -> WATERCOURSE (an observed hydrographic
feature), deliberately kept separate from FeatureType.DRAINAGE (a
PR15-detected valley line, which is a potential-flow-path inference
from a TIN, not evidence that water is actually present there).
Conflating the two would let an automated terrain analysis silently
imply a watercourse exists.

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

DRAINAGE_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Natural drainage (observed, not inferred -- see module docstring)
    #
    FeatureCodeDefinition(
        code="RIO",
        name="River",
        feature_type=FeatureType.WATERCOURSE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="HIDROGRAFIA",
    ),
    FeatureCodeDefinition(
        code="QUEBRADA",
        name="Stream",
        feature_type=FeatureType.WATERCOURSE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="HIDROGRAFIA",
    ),
    FeatureCodeDefinition(
        code="CAÑO",
        name="Creek",
        feature_type=FeatureType.WATERCOURSE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="HIDROGRAFIA",
    ),
    FeatureCodeDefinition(
        code="ARROYO",
        name="Creek",
        feature_type=FeatureType.WATERCOURSE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="HIDROGRAFIA",
    ),
    #
    # Water bodies
    #
    FeatureCodeDefinition(
        code="LAGUNA",
        name="Lagoon",
        feature_type=FeatureType.WATERBODY,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="HIDROGRAFIA",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="LAGO",
        name="Lake",
        feature_type=FeatureType.WATERBODY,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="HIDROGRAFIA",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="EMBALSE",
        name="Reservoir",
        feature_type=FeatureType.WATERBODY,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="OBRAS_HIDRAULICAS",
        closed=True,
    ),
    #
    # Artificial channels
    #
    FeatureCodeDefinition(
        code="CANAL",
        name="Channel",
        feature_type=FeatureType.CHANNEL,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    FeatureCodeDefinition(
        code="ACEQUIA",
        name="Irrigation Canal",
        feature_type=FeatureType.CHANNEL,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    FeatureCodeDefinition(
        code="ZANJA",
        name="Ditch",
        feature_type=FeatureType.CHANNEL,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    FeatureCodeDefinition(
        code="CUNETA",
        name="Road Ditch",
        feature_type=FeatureType.CHANNEL,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    #
    # Sewer network
    #
    FeatureCodeDefinition(
        code="ALCANTARILLA",
        name="Culvert",
        feature_type=FeatureType.SEWER_LINE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="BOX",
        name="Box Culvert",
        feature_type=FeatureType.SEWER_LINE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="TUBERIA",
        name="Drain Pipe",
        feature_type=FeatureType.SEWER_LINE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="COLECTOR",
        name="Collector Sewer",
        feature_type=FeatureType.SEWER_LINE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    #
    # Inspection structures
    #
    FeatureCodeDefinition(
        code="POZO",
        name="Inspection Manhole",
        feature_type=FeatureType.MANHOLE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="MH",
        name="Manhole",
        feature_type=FeatureType.MANHOLE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="SUMIDERO",
        name="Storm Drain Inlet",
        feature_type=FeatureType.STORM_DRAIN_INLET,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="REJILLA",
        name="Drain Grate",
        feature_type=FeatureType.STORM_DRAIN_INLET,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="CAMARA_DRENAJE",
        name="Inspection Chamber",
        feature_type=FeatureType.INSPECTION_CHAMBER,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
        aliases=("CAMARA",),
    ),
    #
    # Hydraulic structures
    #
    FeatureCodeDefinition(
        code="VERTEDERO",
        name="Spillway",
        feature_type=FeatureType.SPILLWAY,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="OBRAS_HIDRAULICAS",
    ),
    FeatureCodeDefinition(
        code="DIQUE",
        name="Levee",
        feature_type=FeatureType.DIKE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="OBRAS_HIDRAULICAS",
    ),
    FeatureCodeDefinition(
        code="REPRESA",
        name="Dam",
        feature_type=FeatureType.DAM,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="OBRAS_HIDRAULICAS",
    ),
)

__all__ = [
    "DRAINAGE_CODES",
]
