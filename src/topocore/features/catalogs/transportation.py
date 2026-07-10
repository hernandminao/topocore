"""
topocore.features.catalogs.transportation
=========================================

Transportation feature catalog.

Contains common field codes used in road, highway, urban and
transportation corridor surveys.

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

TRANSPORTATION_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Road centerlines
    #
    FeatureCodeDefinition(
        code="EJE",
        name="Road Centerline",
        geometry_type=FeatureGeometryType.LINE,
        layer="EJES",
    ),
    FeatureCodeDefinition(
        code="EJEVIAL",
        name="Road Centerline",
        geometry_type=FeatureGeometryType.LINE,
        layer="EJES",
    ),
    FeatureCodeDefinition(
        code="CL",
        name="Centerline",
        geometry_type=FeatureGeometryType.LINE,
        layer="EJES",
    ),
    #
    # Pavement
    #
    FeatureCodeDefinition(
        code="BORDE",
        name="Pavement Edge",
        geometry_type=FeatureGeometryType.LINE,
        layer="VIAS",
    ),
    FeatureCodeDefinition(
        code="BORDEPAV",
        name="Pavement Edge",
        geometry_type=FeatureGeometryType.LINE,
        layer="VIAS",
    ),
    FeatureCodeDefinition(
        code="PAV",
        name="Pavement",
        geometry_type=FeatureGeometryType.LINE,
        layer="VIAS",
    ),
    FeatureCodeDefinition(
        code="CALZADA",
        name="Roadway",
        geometry_type=FeatureGeometryType.LINE,
        layer="VIAS",
    ),
    #
    # Curbs
    #
    FeatureCodeDefinition(
        code="SARDINEL",
        name="Curb",
        geometry_type=FeatureGeometryType.LINE,
        layer="BORDILLOS",
    ),
    FeatureCodeDefinition(
        code="BORDILLO",
        name="Curb",
        geometry_type=FeatureGeometryType.LINE,
        layer="BORDILLOS",
    ),
    FeatureCodeDefinition(
        code="CURB",
        name="Curb",
        geometry_type=FeatureGeometryType.LINE,
        layer="BORDILLOS",
    ),
    #
    # Sidewalks
    #
    FeatureCodeDefinition(
        code="ANDEN",
        name="Sidewalk",
        geometry_type=FeatureGeometryType.LINE,
        layer="ANDENES",
    ),
    FeatureCodeDefinition(
        code="ACERA",
        name="Sidewalk",
        geometry_type=FeatureGeometryType.LINE,
        layer="ANDENES",
    ),
    #
    # Medians
    #
    FeatureCodeDefinition(
        code="SEPARADOR",
        name="Median",
        geometry_type=FeatureGeometryType.LINE,
        layer="SEPARADORES",
    ),
    #
    # Shoulders
    #
    FeatureCodeDefinition(
        code="BERMA",
        name="Road Shoulder",
        geometry_type=FeatureGeometryType.LINE,
        layer="BERMAS",
    ),
    #
    # Slopes
    #
    FeatureCodeDefinition(
        code="TALUD",
        name="Slope",
        geometry_type=FeatureGeometryType.LINE,
        layer="TALUDES",
    ),
    FeatureCodeDefinition(
        code="CORONA",
        name="Top of Slope",
        geometry_type=FeatureGeometryType.LINE,
        layer="TALUDES",
    ),
    FeatureCodeDefinition(
        code="PIE",
        name="Toe of Slope",
        geometry_type=FeatureGeometryType.LINE,
        layer="TALUDES",
    ),
    #
    # Ditches
    #
    FeatureCodeDefinition(
        code="CUNETA",
        name="Road Ditch",
        geometry_type=FeatureGeometryType.LINE,
        layer="DRENAJE",
    ),
    #
    # Guard rails
    #
    FeatureCodeDefinition(
        code="BARRERA",
        name="Guard Rail",
        geometry_type=FeatureGeometryType.LINE,
        layer="SEGURIDAD",
    ),
    FeatureCodeDefinition(
        code="DEFENSA",
        name="Guard Rail",
        geometry_type=FeatureGeometryType.LINE,
        layer="SEGURIDAD",
    ),
    #
    # Bridges
    #
    FeatureCodeDefinition(
        code="PUENTE",
        name="Bridge",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PUENTES",
        closed=True,
    ),
    #
    # Tunnels
    #
    FeatureCodeDefinition(
        code="TUNEL",
        name="Tunnel",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="TUNELES",
        closed=True,
    ),
    #
    # Traffic islands
    #
    FeatureCodeDefinition(
        code="ISLETA",
        name="Traffic Island",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ISLETAS",
        closed=True,
    ),
    #
    # Roundabouts
    #
    FeatureCodeDefinition(
        code="GLORIETA",
        name="Roundabout",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="GLORIETAS",
        closed=True,
    ),
)

__all__ = [
    "TRANSPORTATION_CODES",
]
