"""
topocore.features.catalogs.catalog_audit
===========================================

Architectural validator + report for the full TopoCore feature-code
catalog. More than statistics: this is what a future contributor
runs after adding 20 new codes for a company/country catalog, to
know immediately whether the model is still consistent.

Checks performed
-----------------
1. Code/alias uniqueness (defense in depth -- FeatureCodeRegistry
   already enforces this at registration time, but this works on a
   raw definitions list too, without needing a registry).
2. Non-empty layer.
3. ``closed`` consistency: only valid for ``FeatureGeometryType.POLYGON``.
4. Geometry validity, delegated to ``catalogs._validation``.
5. FeatureType -> FeatureCategory consistency: a given FeatureType
   must map to exactly one FeatureCategory across the *entire*
   catalog, never varying by which code or catalog file declared it.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from topocore.features.catalogs._pending import PENDING_CODES
from topocore.features.catalogs._validation import (
    CatalogGeometryError,
    validate_definition_geometry,
)
from topocore.features.feature_codes import FeatureCodeDefinition, FeatureGeometryType
from topocore.features.models import FeatureCategory, FeatureType

_GEOMETRY_LABELS: dict[FeatureGeometryType, str] = {
    FeatureGeometryType.POINT: "POINT",
    FeatureGeometryType.SYMBOL: "SYMBOL",
    FeatureGeometryType.LINE: "LINE",
    FeatureGeometryType.POLYGON: "POLYGON",
    FeatureGeometryType.GROUND: "GROUND",
}


@dataclass(frozen=True, slots=True)
class AuditViolation:
    code: str
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class CatalogAuditReport:
    total_codes: int
    total_aliases: int
    total_feature_types_used: int
    total_feature_types_declared: int
    total_categories: int
    by_category: Mapping[str, int]
    by_geometry: Mapping[str, int]
    codes_per_feature_type: Mapping[str, int]
    top_feature_types: tuple[tuple[str, int], ...]
    aliases: Mapping[str, str]
    pending: tuple[str, ...]
    violations: tuple[AuditViolation, ...]

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    @property
    def coverage_percent(self) -> float:
        if self.total_feature_types_declared == 0:
            return 0.0
        return round(100 * self.total_feature_types_used / self.total_feature_types_declared, 1)

    @property
    def average_codes_per_type(self) -> float:
        if self.total_feature_types_used == 0:
            return 0.0
        return round(self.total_codes / self.total_feature_types_used, 2)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_codes": self.total_codes,
            "total_aliases": self.total_aliases,
            "total_feature_types_used": self.total_feature_types_used,
            "total_feature_types_declared": self.total_feature_types_declared,
            "total_categories": self.total_categories,
            "coverage_percent": self.coverage_percent,
            "average_codes_per_type": self.average_codes_per_type,
            "by_category": dict(self.by_category),
            "by_geometry": dict(self.by_geometry),
            "codes_per_feature_type": dict(self.codes_per_feature_type),
            "top_feature_types": list(self.top_feature_types),
            "aliases": dict(self.aliases),
            "pending": list(self.pending),
            "violations": [(v.code, v.kind, v.message) for v in self.violations],
            "passed": self.passed,
        }

    def __str__(self) -> str:
        lines = [
            "TopoCore Catalog Audit",
            "",
            f"Survey codes...................{self.total_codes}",
            f"Aliases.........................{self.total_aliases}",
            f"FeatureType (declared)..........{self.total_feature_types_declared}",
            f"FeatureType (used by a code)....{self.total_feature_types_used}",
            f"FeatureCategory..................{self.total_categories}",
            f"Coverage.........................{self.coverage_percent}%",
            f"Average codes/type...............{self.average_codes_per_type}",
            "",
            "By category:",
        ]
        for category, count in sorted(self.by_category.items()):
            lines.append(f"  {category:.<24}{count}")

        lines.append("")
        lines.append("By geometry:")
        for geometry, count in sorted(self.by_geometry.items()):
            lines.append(f"  {geometry:.<24}{count}")

        lines.append("")
        lines.append("Top FeatureType by code count:")
        for feature_type, count in self.top_feature_types[:10]:
            lines.append(f"  {feature_type:.<24}{count}")

        lines.append("")
        lines.append("Pending:")
        if self.pending:
            for code in self.pending:
                lines.append(f"  {code}")
        else:
            lines.append("  None")

        lines.append("")
        lines.append("Violations:")
        if self.violations:
            for v in self.violations:
                lines.append(f"  [{v.kind}] {v.code}: {v.message}")
        else:
            lines.append("  None")

        lines.append("")
        lines.append(f"Status: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def run_audit(definitions: tuple[FeatureCodeDefinition, ...]) -> CatalogAuditReport:
    violations: list[AuditViolation] = []

    # 1. code/alias uniqueness
    seen_codes: dict[str, FeatureCodeDefinition] = {}
    seen_aliases: dict[str, str] = {}
    aliases: dict[str, str] = {}

    for d in definitions:
        key = d.code.upper()
        if key in seen_codes and seen_codes[key] != d:
            violations.append(
                AuditViolation(
                    d.code,
                    "duplicate_code",
                    f"Code '{d.code}' declared more than once with different content.",
                )
            )
        seen_codes[key] = d

        for alias in d.aliases:
            alias_key = alias.upper()
            if alias_key in seen_aliases and seen_aliases[alias_key] != d.code:
                violations.append(
                    AuditViolation(
                        d.code,
                        "duplicate_alias",
                        f"Alias '{alias}' already points to '{seen_aliases[alias_key]}'.",
                    )
                )
            seen_aliases[alias_key] = d.code
            aliases[alias] = d.code

    # 2. non-empty layer
    for d in definitions:
        if not d.layer or not d.layer.strip():
            violations.append(AuditViolation(d.code, "empty_layer", "layer must be a non-empty string."))

    # 3. closed consistency
    for d in definitions:
        if d.closed and d.geometry_type != FeatureGeometryType.POLYGON:
            violations.append(
                AuditViolation(
                    d.code,
                    "invalid_closed",
                    f"closed=True is only valid for POLYGON, got {d.geometry_type.value}.",
                )
            )

    # 4. geometry validity
    for d in definitions:
        try:
            validate_definition_geometry(d)
        except CatalogGeometryError as exc:
            violations.append(AuditViolation(d.code, "geometry", str(exc)))

    # 5. FeatureType -> FeatureCategory consistency across the whole catalog
    type_to_categories: dict[FeatureType, set[FeatureCategory]] = {}
    for d in definitions:
        if d.feature_type is None:
            continue
        type_to_categories.setdefault(d.feature_type, set()).add(d.category)

    for feature_type, categories in type_to_categories.items():
        if len(categories) > 1:
            names = ", ".join(sorted(c.value for c in categories))
            violations.append(
                AuditViolation(
                    feature_type.value,
                    "inconsistent_category",
                    f"FeatureType.{feature_type.name} maps to multiple categories: {names}.",
                )
            )

    # statistics
    by_category = Counter(d.category.value for d in definitions)
    by_geometry = Counter(_GEOMETRY_LABELS[d.geometry_type] for d in definitions)
    codes_per_type = Counter(d.feature_type.value for d in definitions if d.feature_type is not None)
    top_types = tuple(sorted(codes_per_type.items(), key=lambda kv: (-kv[1], kv[0])))

    return CatalogAuditReport(
        total_codes=len(definitions),
        total_aliases=len(aliases),
        total_feature_types_used=len(codes_per_type),
        total_feature_types_declared=len(list(FeatureType)),
        total_categories=len(list(FeatureCategory)),
        by_category=dict(by_category),
        by_geometry=dict(by_geometry),
        codes_per_feature_type=dict(codes_per_type),
        top_feature_types=top_types,
        aliases=aliases,
        pending=tuple(p.code for p in PENDING_CODES),
        violations=tuple(violations),
    )


__all__ = ["AuditViolation", "CatalogAuditReport", "run_audit"]
