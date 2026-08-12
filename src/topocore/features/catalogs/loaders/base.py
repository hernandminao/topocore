"""
topocore.features.catalogs.loaders.base
========================================

Shared conversion and validation for external feature-code catalogs.

External catalogs may define their own field codes, aliases, names,
layers, and geometry declarations, but they do not extend TopoCore's
semantic ontology dynamically. Every non-ground entry must map to an
existing ``FeatureType`` and ``FeatureCategory``.

Two error-collecting stages exist and are merged into one report:

1. Raw extraction (``build_raw_entry_from_mapping``, used by both
   ``json_loader``/``yaml_loader``): strict type checks on the
   parsed JSON/YAML structure itself -- e.g. ``closed`` must be an
   actual boolean, not a string TopoCore then guesses at. Rejecting
   here instead of coercing (``bool("false")`` is ``True`` in
   Python) is what prevents a silently wrong catalog.
2. Semantic parsing (``parse_entries``): enum resolution and the
   geometry invariant.

Both stages report every problem they find, not just the first, and
a loader that hits problems in both stages combines them into a
single ``ExternalCatalogError``.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from topocore.features.catalogs._validation import (
    validate_definition_geometry,
)
from topocore.features.feature_codes import (
    FeatureCodeDefinition,
    FeatureGeometryType,
)
from topocore.features.models import FeatureCategory, FeatureType

from .schema import RawCatalogEntry


@dataclass(frozen=True, slots=True)
class CatalogLoadIssue:
    """One validation problem found in an external catalog."""

    index: int
    code: str
    message: str


class ExternalCatalogError(ValueError):
    """
    Raised when an external catalog contains invalid definitions.

    All detected entry-level problems -- from raw extraction and/or
    semantic parsing -- are reported together so users can correct
    the catalog in a single pass.
    """

    def __init__(self, issues: Iterable[CatalogLoadIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        details = "\n".join(f"  [{issue.index}] {issue.code or '<empty>'}: {issue.message}" for issue in self.issues)
        return f"External feature-code catalog contains {len(self.issues)} error(s):\n{details}"


# ---------------------------------------------------------------------
# Stage 1: raw extraction, strict types -- shared by json_loader and
# yaml_loader so their behavior can't silently drift apart.
# ---------------------------------------------------------------------


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise TypeError(f"'{key}' must be a string, got {type(value).__name__} ({value!r}).")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"'{key}' must be a string or null/omitted, got {type(value).__name__} ({value!r}).")
    return value


def _require_bool(data: dict[str, Any], key: str, *, default: bool = False) -> bool:
    if key not in data or data[key] is None:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(
            f"'{key}' must be a boolean (true/false), got {type(value).__name__} ({value!r}). "
            f'No implicit coercion is applied -- a string like "false" is truthy in Python '
            f"and would silently mean the opposite of what it looks like."
        )
    return value


def _require_alias_list(data: dict[str, Any]) -> tuple[str, ...]:
    raw = data.get("aliases", [])
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(f"'aliases' must be a list, got {type(raw).__name__}.")

    aliases: list[str] = []
    for position, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(f"aliases[{position}] must be a string, got {type(item).__name__} ({item!r}).")
        stripped = item.strip()
        if stripped:
            aliases.append(stripped)
    return tuple(aliases)


def build_raw_entry_from_mapping(data: dict[str, Any]) -> RawCatalogEntry:
    """
    Strictly-typed construction of a ``RawCatalogEntry`` from one
    parsed JSON/YAML mapping. Shared by ``json_loader``/
    ``yaml_loader`` so both formats validate identically instead of
    maintaining two near-duplicate implementations.

    No ``index`` parameter here: entry-index bookkeeping for
    ``CatalogLoadIssue`` is entirely the caller's job (its own
    ``enumerate()`` loop), not this function's -- it was previously
    threaded through unused (SonarQube python:S1172).

    Raises
    ------
    ValueError
        If any field has the wrong type. No ``str()``/``bool()``
        coercion is applied anywhere -- a numeric ``code`` or a
        string-valued ``closed`` is a catalog-authoring mistake, not
        something to silently paper over.
    """
    return RawCatalogEntry(
        code=_require_str(data, "code"),
        name=_require_str(data, "name"),
        geometry_type=_require_str(data, "geometry_type"),
        feature_type=_optional_str(data, "feature_type"),
        category=_require_str(data, "category"),
        layer=_require_str(data, "layer"),
        closed=_require_bool(data, "closed", default=False),
        aliases=_require_alias_list(data),
    )


# ---------------------------------------------------------------------
# Stage 2: semantic parsing -- enum resolution + geometry invariant.
# ---------------------------------------------------------------------


def _parse_geometry_type(value: str) -> FeatureGeometryType:
    """
    Value-based lookup (lowercase), matching `_parse_feature_type`/
    `_parse_category` -- a catalog author writes `line`/`fence`/
    `building` throughout, one convention, not `LINE` here and
    `fence` there.
    """
    normalized = value.strip().lower()

    try:
        return FeatureGeometryType(normalized)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in FeatureGeometryType))
        raise ValueError(f"Unknown geometry_type '{value}'. Expected one of: {allowed}.") from exc


def _parse_feature_type(value: str | None) -> FeatureType | None:
    if value is None or not value.strip():
        return None

    normalized = value.strip().lower()

    try:
        return FeatureType(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Unknown feature_type '{value}'. External catalogs remap "
            "field codes onto TopoCore's existing ontology; they do not "
            "introduce new FeatureType values."
        ) from exc


def _parse_category(value: str) -> FeatureCategory:
    normalized = value.strip().lower()

    try:
        return FeatureCategory(normalized)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in FeatureCategory))
        raise ValueError(f"Unknown category '{value}'. Expected one of: {allowed}.") from exc


def _parse_entry(entry: RawCatalogEntry) -> FeatureCodeDefinition:
    code = entry.code.strip()
    name = entry.name.strip()
    layer = entry.layer.strip()

    if not code:
        raise ValueError("code must not be empty.")
    if not name:
        raise ValueError(f"Code '{code}' must declare a non-empty name.")
    if not layer:
        raise ValueError(f"Code '{code}' must declare a non-empty layer.")

    geometry_type = _parse_geometry_type(entry.geometry_type)
    feature_type = _parse_feature_type(entry.feature_type)
    category = _parse_category(entry.category)

    definition = FeatureCodeDefinition(
        code=code,
        name=name,
        geometry_type=geometry_type,
        layer=layer,
        feature_type=feature_type,
        category=category,
        closed=entry.closed,
        aliases=entry.aliases,
    )

    validate_definition_geometry(definition)

    return definition


def parse_entries(
    entries: Iterable[RawCatalogEntry],
    *,
    prior_issues: Iterable[CatalogLoadIssue] = (),
) -> tuple[FeatureCodeDefinition, ...]:
    """
    Convert raw external entries into validated TopoCore definitions.

    Parameters
    ----------
    prior_issues
        Issues already collected by an earlier stage (e.g. raw
        extraction type errors) -- merged into the same
        ``ExternalCatalogError`` instead of being reported separately,
        so a catalog author sees every problem in the file at once.
    """
    definitions: list[FeatureCodeDefinition] = []
    issues: list[CatalogLoadIssue] = list(prior_issues)

    for index, entry in enumerate(entries, start=1):
        try:
            definitions.append(_parse_entry(entry))
        except ValueError as exc:
            issues.append(CatalogLoadIssue(index=index, code=entry.code, message=str(exc)))

    if issues:
        raise ExternalCatalogError(issues)

    return tuple(definitions)


__all__ = [
    "CatalogLoadIssue",
    "ExternalCatalogError",
    "build_raw_entry_from_mapping",
    "parse_entries",
]
