"""
topocore.features.catalogs.cadastre
====================================

Cadastre and property boundary feature catalog.

Note
----
HITO and PI live here (a cadastral-survey convention: boundary
markers and alignment points are commonly captured during property
surveys) but get `category=CONTROL`, matching MOJON's category in
control.py -- boundary monuments and alignment points are control
concepts, not property-parcel concepts, regardless of which catalog
file declares their field code. Same principle already established
for TALUD/CORONA/PIE in transportation.py.

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

CADASTRE_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Property boundaries
    #
    FeatureCodeDefinition(
        code="LOTE",
        name="Property",
        feature_type=FeatureType.PARCEL,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PREDIOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PREDIO",
        name="Property",
        feature_type=FeatureType.PARCEL,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PREDIOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="PARCELA",
        name="Parcel",
        feature_type=FeatureType.PARCEL,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PREDIOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="FINCA",
        name="Farm",
        feature_type=FeatureType.PARCEL,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PREDIOS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="MANZANA",
        name="Block",
        feature_type=FeatureType.BLOCK,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="MANZANAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="SECTOR",
        name="Sector",
        feature_type=FeatureType.ZONE,
        category=FeatureCategory.CADASTRE,
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
        feature_type=FeatureType.BOUNDARY,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.LINE,
        layer="LINDEROS",
    ),
    FeatureCodeDefinition(
        code="LIMITE",
        name="Boundary",
        feature_type=FeatureType.BOUNDARY,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.LINE,
        layer="LINDEROS",
    ),
    FeatureCodeDefinition(
        code="DESLINDE",
        name="Boundary",
        feature_type=FeatureType.BOUNDARY,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.LINE,
        layer="LINDEROS",
    ),
    FeatureCodeDefinition(
        code="SERVIDUMBRE",
        name="Easement",
        feature_type=FeatureType.EASEMENT,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.LINE,
        layer="SERVIDUMBRES",
    ),
    FeatureCodeDefinition(
        code="RETIRO",
        name="Setback",
        feature_type=FeatureType.SETBACK,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.LINE,
        layer="RETIROS",
    ),
    #
    # Boundary monuments / alignment points -- CONTROL category
    # despite living in this file, see module docstring.
    #
    FeatureCodeDefinition(
        code="HITO",
        name="Boundary Marker",
        feature_type=FeatureType.BOUNDARY_MONUMENT,
        category=FeatureCategory.CONTROL,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    FeatureCodeDefinition(
        code="PI",
        name="Intersection Point",
        feature_type=FeatureType.ALIGNMENT_PI,
        category=FeatureCategory.CONTROL,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="CONTROL",
    ),
    #
    # Administrative
    #
    FeatureCodeDefinition(
        code="MUNICIPIO",
        name="Municipality Boundary",
        feature_type=FeatureType.ADMINISTRATIVE_BOUNDARY,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ADMINISTRATIVO",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="VEREDA",
        name="Village",
        feature_type=FeatureType.ADMINISTRATIVE_BOUNDARY,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ADMINISTRATIVO",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="BARRIO",
        name="Neighborhood",
        feature_type=FeatureType.ADMINISTRATIVE_BOUNDARY,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ADMINISTRATIVO",
        closed=True,
    ),
    #
    # Rights of way / zones
    #
    FeatureCodeDefinition(
        code="FAJA",
        name="Right of Way",
        feature_type=FeatureType.RIGHT_OF_WAY,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="SERVIDUMBRES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="ZONA",
        name="Zone",
        feature_type=FeatureType.ZONE,
        category=FeatureCategory.CADASTRE,
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
        feature_type=FeatureType.REFERENCE_POINT,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="REFERENCIAS",
    ),
    FeatureCodeDefinition(
        code="REFERENCIA",
        name="Reference Point",
        feature_type=FeatureType.REFERENCE_POINT,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="REFERENCIAS",
    ),
    FeatureCodeDefinition(
        code="CLAVO",
        name="Survey Nail",
        feature_type=FeatureType.REFERENCE_POINT,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="REFERENCIAS",
    ),
    FeatureCodeDefinition(
        code="ESTACA",
        name="Stake",
        feature_type=FeatureType.REFERENCE_POINT,
        category=FeatureCategory.CADASTRE,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="REFERENCIAS",
    ),
)

__all__ = [
    "CADASTRE_CODES",
]
