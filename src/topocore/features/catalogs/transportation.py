"""
topocore.features.catalogs.transportation
=========================================

Transportation feature catalog.

Note
----
TALUD/CORONA/PIE live in this file (a road-alignment survey
convention -- embankments are commonly captured alongside roads) but
get `category=TERRAIN`, matching FeatureType.EMBANKMENT/
EMBANKMENT_CREST/EMBANKMENT_TOE's category -- an embankment is a
geomorphological feature, not a road structure, regardless of which
catalog file happens to declare its field code. `category` is a
property of `feature_type`, never inferred from catalog location.

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

TRANSPORTATION_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Road centerlines
    #
    FeatureCodeDefinition(
        code="EJE",
        name="Road Centerline",
        feature_type=FeatureType.CENTERLINE,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="EJES",
    ),
    FeatureCodeDefinition(
        code="EJEVIAL",
        name="Road Centerline",
        feature_type=FeatureType.CENTERLINE,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="EJES",
    ),
    FeatureCodeDefinition(
        code="CL",
        name="Centerline",
        feature_type=FeatureType.CENTERLINE,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="EJES",
    ),
    #
    # Pavement
    #
    FeatureCodeDefinition(
        code="BORDE",
        name="Pavement Edge",
        feature_type=FeatureType.PAVEMENT_EDGE,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="VIAS",
    ),
    FeatureCodeDefinition(
        code="BORDEPAV",
        name="Pavement Edge",
        feature_type=FeatureType.PAVEMENT_EDGE,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="VIAS",
    ),
    FeatureCodeDefinition(
        code="PAV",
        name="Pavement",
        feature_type=FeatureType.PAVEMENT_EDGE,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="VIAS",
    ),
    FeatureCodeDefinition(
        code="CALZADA",
        name="Roadway",
        feature_type=FeatureType.PAVEMENT_EDGE,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="VIAS",
    ),
    #
    # Curbs
    #
    FeatureCodeDefinition(
        code="SARDINEL",
        name="Curb",
        feature_type=FeatureType.CURB,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="BORDILLOS",
    ),
    FeatureCodeDefinition(
        code="BORDILLO",
        name="Curb",
        feature_type=FeatureType.CURB,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="BORDILLOS",
    ),
    FeatureCodeDefinition(
        code="CURB",
        name="Curb",
        feature_type=FeatureType.CURB,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="BORDILLOS",
    ),
    #
    # Sidewalks
    #
    FeatureCodeDefinition(
        code="ANDEN",
        name="Sidewalk",
        feature_type=FeatureType.SIDEWALK,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="ANDENES",
    ),
    FeatureCodeDefinition(
        code="ACERA",
        name="Sidewalk",
        feature_type=FeatureType.SIDEWALK,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="ANDENES",
    ),
    #
    # Medians / shoulders
    #
    FeatureCodeDefinition(
        code="SEPARADOR",
        name="Median",
        feature_type=FeatureType.MEDIAN,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="SEPARADORES",
    ),
    FeatureCodeDefinition(
        code="BERMA",
        name="Road Shoulder",
        feature_type=FeatureType.SHOULDER,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="BERMAS",
    ),
    #
    # Slopes -- TERRAIN category despite living in this file, see
    # module docstring.
    #
    FeatureCodeDefinition(
        code="TALUD",
        name="Slope",
        feature_type=FeatureType.EMBANKMENT,
        category=FeatureCategory.TERRAIN,
        geometry_type=FeatureGeometryType.LINE,
        layer="TALUDES",
    ),
    FeatureCodeDefinition(
        code="CORONA",
        name="Top of Slope",
        feature_type=FeatureType.EMBANKMENT_CREST,
        category=FeatureCategory.TERRAIN,
        geometry_type=FeatureGeometryType.LINE,
        layer="TALUDES",
    ),
    FeatureCodeDefinition(
        code="PIE",
        name="Toe of Slope",
        feature_type=FeatureType.EMBANKMENT_TOE,
        category=FeatureCategory.TERRAIN,
        geometry_type=FeatureGeometryType.LINE,
        layer="TALUDES",
    ),
    #
    # Guard rails
    #
    FeatureCodeDefinition(
        code="BARRERA",
        name="Guard Rail",
        feature_type=FeatureType.GUARDRAIL,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="SEGURIDAD",
    ),
    FeatureCodeDefinition(
        code="DEFENSA",
        name="Guard Rail",
        feature_type=FeatureType.GUARDRAIL,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.LINE,
        layer="SEGURIDAD",
    ),
    #
    # Bridges / tunnels
    #
    FeatureCodeDefinition(
        code="PUENTE",
        name="Bridge",
        feature_type=FeatureType.BRIDGE,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="PUENTES",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="TUNEL",
        name="Tunnel",
        feature_type=FeatureType.TUNNEL,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="TUNELES",
        closed=True,
    ),
    #
    # Traffic islands / roundabouts
    #
    FeatureCodeDefinition(
        code="ISLETA",
        name="Traffic Island",
        feature_type=FeatureType.TRAFFIC_ISLAND,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ISLETAS",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="GLORIETA",
        name="Roundabout",
        feature_type=FeatureType.ROUNDABOUT,
        category=FeatureCategory.INFRASTRUCTURE,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="GLORIETAS",
        closed=True,
    ),
)

__all__ = [
    "TRANSPORTATION_CODES",
]
