"""
topocore.features.catalogs.utilities
====================================

Utilities feature catalog.

Note
----
ALCANTARILLADO lives here but gets `category=DRAINAGE` and
`feature_type=SEWER_LINE` -- same physical concept as ALCANTARILLA/
BOX/TUBERIA/COLECTOR in drainage.py (third instance of the
category-vs-catalog-file-location pattern, after TALUD/CORONA/PIE in
transportation.py and HITO/PI in cadastre.py).

REFLECTOR collapses into LIGHT_POLE (confirmed: a flood light is a
lighting fixture, same domain as a street light).

VALVULA/VALVGAS both collapse to VALVE, and CAJA/CAJATELECOM/CAJAAGUA
all collapse to UTILITY_BOX -- "same physical object -> same
FeatureType; service/network -> attribute" rule confirmed several
rounds ago. The `utility_service` distinction (water/gas/telecom/
electric) is not stored on FeatureCodeDefinition -- it belongs to
Feature.attributes once the Survey -> FeatureCollection bridge
exists (a later step); the original `code` alone already preserves
which service each definition came from.

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

UTILITY_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Electrical distribution
    #
    FeatureCodeDefinition(
        code="POSTE",
        name="Utility Pole",
        feature_type=FeatureType.POLE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="POSTEBT",
        name="Low Voltage Pole",
        feature_type=FeatureType.POLE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="POSTEMT",
        name="Medium Voltage Pole",
        feature_type=FeatureType.POLE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="POSTEAT",
        name="High Voltage Pole",
        feature_type=FeatureType.POLE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="TORRE",
        name="Transmission Tower",
        feature_type=FeatureType.TRANSMISSION_TOWER,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="TRANSFORMADOR",
        name="Transformer",
        feature_type=FeatureType.TRANSFORMER,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="SUBESTACION",
        name="Substation",
        feature_type=FeatureType.SUBSTATION,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ELECTRICO",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="LINEABT",
        name="Low Voltage Line",
        feature_type=FeatureType.POWER_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="LINEAMT",
        name="Medium Voltage Line",
        feature_type=FeatureType.POWER_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="LINEAAT",
        name="High Voltage Line",
        feature_type=FeatureType.POWER_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="ELECTRICO",
    ),
    #
    # Public lighting
    #
    FeatureCodeDefinition(
        code="LUMINARIA",
        name="Street Light",
        feature_type=FeatureType.LIGHT_POLE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALUMBRADO",
    ),
    FeatureCodeDefinition(
        code="FAROLA",
        name="Street Light",
        feature_type=FeatureType.LIGHT_POLE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALUMBRADO",
    ),
    FeatureCodeDefinition(
        code="REFLECTOR",
        name="Flood Light",
        feature_type=FeatureType.LIGHT_POLE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALUMBRADO",
    ),
    #
    # Telecommunications
    #
    FeatureCodeDefinition(
        code="TELECOM",
        name="Telecommunications Line",
        feature_type=FeatureType.TELECOM_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="TELECOM",
    ),
    FeatureCodeDefinition(
        code="FIBRA",
        name="Fiber Optic Cable",
        feature_type=FeatureType.TELECOM_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="TELECOM",
    ),
    FeatureCodeDefinition(
        code="CAJATELECOM",
        name="Telecommunications Box",
        feature_type=FeatureType.UTILITY_BOX,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="TELECOM",
    ),
    FeatureCodeDefinition(
        code="ANTENA",
        name="Telecommunications Antenna",
        feature_type=FeatureType.ANTENNA,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="TELECOM",
    ),
    FeatureCodeDefinition(
        code="TORRECEL",
        name="Cell Tower",
        feature_type=FeatureType.ANTENNA,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="TELECOM",
    ),
    #
    # Water supply
    #
    FeatureCodeDefinition(
        code="ACUEDUCTO",
        name="Water Main",
        feature_type=FeatureType.WATER_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="REDAGUA",
        name="Water Pipeline",
        feature_type=FeatureType.WATER_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="VALVULA",
        name="Valve",
        feature_type=FeatureType.VALVE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="HIDRANTE",
        name="Fire Hydrant",
        feature_type=FeatureType.HYDRANT,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="MACROMEDIDOR",
        name="Master Water Meter",
        feature_type=FeatureType.WATER_METER,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="CAJAAGUA",
        name="Water Meter Box",
        feature_type=FeatureType.UTILITY_BOX,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ACUEDUCTO",
    ),
    #
    # Sewer -- DRAINAGE category despite living in this file, see
    # module docstring.
    #
    FeatureCodeDefinition(
        code="ALCANTARILLADO",
        name="Sewer Main",
        feature_type=FeatureType.SEWER_LINE,
        category=FeatureCategory.DRAINAGE,
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    #
    # Gas
    #
    FeatureCodeDefinition(
        code="GAS",
        name="Gas Pipeline",
        feature_type=FeatureType.GAS_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="GAS",
    ),
    FeatureCodeDefinition(
        code="VALVGAS",
        name="Gas Valve",
        feature_type=FeatureType.VALVE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="GAS",
    ),
    FeatureCodeDefinition(
        code="REGULADOR",
        name="Gas Regulator",
        feature_type=FeatureType.GAS_REGULATOR,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="GAS",
    ),
    #
    # Generic underground utilities
    #
    FeatureCodeDefinition(
        code="DUCTO",
        name="Underground Duct",
        feature_type=FeatureType.UTILITY_LINE,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.LINE,
        layer="SERVICIOS",
    ),
    FeatureCodeDefinition(
        code="CAJA",
        name="Utility Box",
        feature_type=FeatureType.UTILITY_BOX,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="SERVICIOS",
    ),
    FeatureCodeDefinition(
        code="CAMARA_SERVICIOS",
        name="Utility Chamber",
        feature_type=FeatureType.UTILITY_CHAMBER,
        category=FeatureCategory.UTILITY,
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="SERVICIOS",
    ),
)

__all__ = [
    "UTILITY_CODES",
]
