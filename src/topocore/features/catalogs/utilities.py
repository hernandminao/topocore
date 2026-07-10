"""
topocore.features.catalogs.utilities
====================================

Utilities feature catalog.

Contains common field codes used in utility surveys, including
electricity, telecommunications, water supply, sewer, gas and
public lighting infrastructure.

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

UTILITY_CODES: tuple[FeatureCodeDefinition, ...] = (
    #
    # Electrical distribution
    #
    FeatureCodeDefinition(
        code="POSTE",
        name="Utility Pole",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="POSTEBT",
        name="Low Voltage Pole",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="POSTEMT",
        name="Medium Voltage Pole",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="POSTEAT",
        name="High Voltage Pole",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="TORRE",
        name="Transmission Tower",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="TRANSFORMADOR",
        name="Transformer",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="SUBESTACION",
        name="Substation",
        geometry_type=FeatureGeometryType.POLYGON,
        layer="ELECTRICO",
        closed=True,
    ),
    FeatureCodeDefinition(
        code="LINEABT",
        name="Low Voltage Line",
        geometry_type=FeatureGeometryType.LINE,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="LINEAMT",
        name="Medium Voltage Line",
        geometry_type=FeatureGeometryType.LINE,
        layer="ELECTRICO",
    ),
    FeatureCodeDefinition(
        code="LINEAAT",
        name="High Voltage Line",
        geometry_type=FeatureGeometryType.LINE,
        layer="ELECTRICO",
    ),
    #
    # Public lighting
    #
    FeatureCodeDefinition(
        code="LUMINARIA",
        name="Street Light",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALUMBRADO",
    ),
    FeatureCodeDefinition(
        code="FAROLA",
        name="Street Light",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALUMBRADO",
    ),
    FeatureCodeDefinition(
        code="REFLECTOR",
        name="Flood Light",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALUMBRADO",
    ),
    #
    # Telecommunications
    #
    FeatureCodeDefinition(
        code="TELECOM",
        name="Telecommunications Line",
        geometry_type=FeatureGeometryType.LINE,
        layer="TELECOM",
    ),
    FeatureCodeDefinition(
        code="FIBRA",
        name="Fiber Optic Cable",
        geometry_type=FeatureGeometryType.LINE,
        layer="TELECOM",
    ),
    FeatureCodeDefinition(
        code="CAJATELECOM",
        name="Telecommunications Box",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="TELECOM",
    ),
    FeatureCodeDefinition(
        code="ANTENA",
        name="Telecommunications Antenna",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="TELECOM",
    ),
    FeatureCodeDefinition(
        code="TORRECEL",
        name="Cell Tower",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="TELECOM",
    ),
    #
    # Water supply
    #
    FeatureCodeDefinition(
        code="ACUEDUCTO",
        name="Water Main",
        geometry_type=FeatureGeometryType.LINE,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="REDAGUA",
        name="Water Pipeline",
        geometry_type=FeatureGeometryType.LINE,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="VALVULA",
        name="Valve",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="HIDRANTE",
        name="Fire Hydrant",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="MACROMEDIDOR",
        name="Master Water Meter",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ACUEDUCTO",
    ),
    FeatureCodeDefinition(
        code="CAJAAGUA",
        name="Water Meter Box",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ACUEDUCTO",
    ),
    #
    # Sewer
    #
    FeatureCodeDefinition(
        code="ALCANTARILLADO",
        name="Sewer Main",
        geometry_type=FeatureGeometryType.LINE,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="POZO",
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
        code="CAMARA",
        name="Inspection Chamber",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    FeatureCodeDefinition(
        code="REJILLA",
        name="Drain Grate",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="ALCANTARILLADO",
    ),
    #
    # Gas
    #
    FeatureCodeDefinition(
        code="GAS",
        name="Gas Pipeline",
        geometry_type=FeatureGeometryType.LINE,
        layer="GAS",
    ),
    FeatureCodeDefinition(
        code="VALVGAS",
        name="Gas Valve",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="GAS",
    ),
    FeatureCodeDefinition(
        code="REGULADOR",
        name="Gas Regulator",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="GAS",
    ),
    #
    # Generic underground utilities
    #
    FeatureCodeDefinition(
        code="DUCTO",
        name="Underground Duct",
        geometry_type=FeatureGeometryType.LINE,
        layer="SERVICIOS",
    ),
    FeatureCodeDefinition(
        code="CAJA",
        name="Utility Box",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="SERVICIOS",
    ),
    FeatureCodeDefinition(
        code="CAMARA",
        name="Utility Chamber",
        geometry_type=FeatureGeometryType.SYMBOL,
        layer="SERVICIOS",
    ),
)

__all__ = [
    "UTILITY_CODES",
]