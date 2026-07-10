"""
topocore.features.feature_codes
==================================

The field-code dictionary: maps a survey code (e.g. ``"CERCA"``,
``"ARBOL"``) to what kind of geometry it should become and which CAD
layer it belongs to.

This is the same concept as Civil3D's Description Key Sets or
Carlson's linework code library, adapted to TopoCore. It ships with
a small default dictionary of common Spanish-language topographic
field codes; projects are expected to extend or replace it via
``FeatureCodeRegistry.register``.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeatureGeometryType(Enum):
    """
    What kind of geometry a feature code produces.
    """

    #: An isolated point, not connected to others sharing the code.
    POINT = "point"

    #: A point rendered as a CAD block/symbol (e.g. a tree, a pole).
    SYMBOL = "symbol"

    #: Consecutive same-code points connected into a polyline.
    LINE = "line"

    #: Consecutive same-code points connected into a closed polyline.
    POLYGON = "polygon"


@dataclass(frozen=True, slots=True)
class FeatureCodeDefinition:
    """
    Describes what a single field code means.

    Parameters
    ----------
    code
        The base code, without any trailing sequence digits (see
        ``feature_builder``).
    name
        Human-readable name.
    geometry_type
        What kind of geometry this code produces.
    layer
        Destination CAD/GIS layer name.
    closed
        For ``LINE``/``POLYGON`` codes: whether the resulting shape
        should always be closed (e.g. a building footprint), even if
        the first and last surveyed points don't coincide exactly.
    """

    code: str
    name: str
    geometry_type: FeatureGeometryType
    layer: str
    closed: bool = False


class FeatureCodeRegistry:
    """
    A lookup table from base code to ``FeatureCodeDefinition``.
    """

    __slots__ = ("_definitions",)

    def __init__(
        self,
        definitions: dict[str, FeatureCodeDefinition] | None = None,
    ) -> None:
        self._definitions: dict[str, FeatureCodeDefinition] = dict(
            definitions or {}
        )

    @classmethod
    def default(cls) -> FeatureCodeRegistry:
        """
        Return a registry pre-loaded with common topographic codes.
        """
        return cls(dict(_DEFAULT_DEFINITIONS))

    def get(self, code: str) -> FeatureCodeDefinition | None:
        """
        Look up a code, case-insensitively. Returns ``None`` if the
        code is not registered.
        """
        return self._definitions.get(code.upper())

    def register(self, definition: FeatureCodeDefinition) -> None:
        """
        Add or replace a code definition.
        """
        self._definitions[definition.code.upper()] = definition

    def __contains__(self, code: str) -> bool:
        return code.upper() in self._definitions

    def __len__(self) -> int:
        return len(self._definitions)


_DEFAULT_DEFINITIONS: dict[str, FeatureCodeDefinition] = {
    definition.code: definition
    for definition in (
        FeatureCodeDefinition(
            "CERCA", "Cerca", FeatureGeometryType.LINE, "CERCAS",
        ),
        FeatureCodeDefinition(
            "MURO", "Muro de contención", FeatureGeometryType.LINE, "MUROS",
        ),
        FeatureCodeDefinition(
            "BORDE", "Borde de vía", FeatureGeometryType.LINE, "VIAS",
        ),
        FeatureCodeDefinition(
            "SARDINEL", "Sardinel / bordillo", FeatureGeometryType.LINE,
            "VIAS",
        ),
        FeatureCodeDefinition(
            "EJE", "Eje vial", FeatureGeometryType.LINE, "EJES",
        ),
        FeatureCodeDefinition(
            "CANAL", "Canal de drenaje", FeatureGeometryType.LINE,
            "DRENAJE",
        ),
        FeatureCodeDefinition(
            "EDIF", "Edificación", FeatureGeometryType.POLYGON,
            "EDIFICACIONES", closed=True,
        ),
        FeatureCodeDefinition(
            "LOTE", "Lindero de lote", FeatureGeometryType.POLYGON,
            "PREDIOS", closed=True,
        ),
        FeatureCodeDefinition(
            "ARBOL", "Árbol", FeatureGeometryType.SYMBOL, "VEGETACION",
        ),
        FeatureCodeDefinition(
            "POSTE", "Poste de servicios", FeatureGeometryType.SYMBOL,
            "SERVICIOS",
        ),
        FeatureCodeDefinition(
            "POZO", "Pozo de inspección", FeatureGeometryType.SYMBOL,
            "DRENAJE",
        ),
        FeatureCodeDefinition(
            "BM", "Punto de control / mojón", FeatureGeometryType.SYMBOL,
            "CONTROL",
        ),
    )
}


__all__ = [
    "FeatureGeometryType",
    "FeatureCodeDefinition",
    "FeatureCodeRegistry",
]
