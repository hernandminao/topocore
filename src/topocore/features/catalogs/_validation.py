"""
topocore.features.catalogs._validation
=========================================

Mechanical invariant check: every FeatureCodeDefinition's declared
geometry, once mapped through FeatureGeometryType -> GeometryType,
must be a member of _EXPECTED_GEOMETRY[definition.feature_type].

Explicit validation / test utility -- never run as an import-time
side effect, so `import topocore.features` cannot fail due to
catalog data issues.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterable

from topocore.features.feature_codes import (
    CATALOG_TO_MODEL_GEOMETRY,
    FeatureCodeDefinition,
    FeatureGeometryType,
)
from topocore.features.models import _EXPECTED_GEOMETRY


class CatalogGeometryError(ValueError):
    """Raised when a catalog definition's geometry contradicts _EXPECTED_GEOMETRY."""


def validate_definition_geometry(definition: FeatureCodeDefinition) -> None:
    """
    Raises
    ------
    CatalogGeometryError
        If ``definition.geometry_type is GROUND`` but ``feature_type``
        is set (GROUND codes never produce a Feature); if
        ``feature_type is None`` for any non-GROUND code (every code
        that produces a Feature must declare its type); or if
        ``definition.geometry_type`` maps to a ``GeometryType`` not
        allowed for ``definition.feature_type`` in ``_EXPECTED_GEOMETRY``.
    """
    if definition.geometry_type is FeatureGeometryType.GROUND:
        if definition.feature_type is not None:
            raise CatalogGeometryError(
                f"Ground code '{definition.code}' must not declare a feature_type; "
                "GROUND survey points feed terrain modelling and do not produce Features."
            )
        return

    if definition.feature_type is None:
        raise CatalogGeometryError(
            f"Code '{definition.code}' must declare a feature_type because "
            f"geometry_type={definition.geometry_type.value} produces a Feature."
        )

    expected_model_geometry = CATALOG_TO_MODEL_GEOMETRY[definition.geometry_type]
    allowed = _EXPECTED_GEOMETRY.get(definition.feature_type)

    if allowed is None:
        raise CatalogGeometryError(
            f"Code '{definition.code}' declares feature_type={definition.feature_type}, "
            f"which has no entry in _EXPECTED_GEOMETRY at all."
        )

    if expected_model_geometry not in allowed:
        allowed_names = ", ".join(sorted(g.value for g in allowed))
        raise CatalogGeometryError(
            f"Code '{definition.code}' declares geometry_type={definition.geometry_type.value} "
            f"(-> {expected_model_geometry.value}), but feature_type={definition.feature_type.value} "
            f"only allows [{allowed_names}] per _EXPECTED_GEOMETRY."
        )


def validate_all(definitions: Iterable[FeatureCodeDefinition]) -> None:
    """
    Validate every definition; raises on the first violation found.

    Accepts any ``Iterable`` -- tuple, list, generator, or a future
    registry view -- not coupled to a specific container type.
    """
    for definition in definitions:
        validate_definition_geometry(definition)


__all__ = ["CatalogGeometryError", "validate_all", "validate_definition_geometry"]
