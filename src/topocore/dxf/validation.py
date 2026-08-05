from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from topocore.dxf.exceptions import DXFValidationError
from topocore.dxf.xdata import non_scalar_attribute_keys
from topocore.features.models import Feature, FeatureType, GeometryType


class ValidationSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    feature_id: int | None
    severity: ValidationSeverity
    code: str
    message: str


class DXFValidator:
    def validate(self, feature: Feature) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if feature.geometry.geometry_type == GeometryType.MESH:
            if feature.geometry.faces is None or feature.geometry.faces.shape[0] == 0:
                issues.append(
                    ValidationIssue(
                        feature.feature_id,
                        ValidationSeverity.ERROR,
                        "DXF001",
                        "MESH geometry has no faces to export.",
                    )
                )

        non_scalar = non_scalar_attribute_keys(dict(feature.attributes))
        if non_scalar:
            issues.append(
                ValidationIssue(
                    feature.feature_id,
                    ValidationSeverity.WARNING,
                    "DXF002",
                    f"Non-scalar attributes will be dropped from XDATA: {non_scalar}.",
                )
            )

        if feature.feature_type == FeatureType.CONTOUR:
            extra = feature.metadata.extra if feature.metadata else {}
            missing = [
                name
                for name, present in (
                    ("elevation", "elevation" in feature.attributes),
                    ("base", "base" in extra),
                    ("interval", "interval" in extra),
                )
                if not present
            ]
            if missing:
                issues.append(
                    ValidationIssue(
                        feature.feature_id,
                        ValidationSeverity.WARNING,
                        "DXF003",
                        f"Contour is missing {missing} metadata; it will be placed on "
                        "the neutral TOPO_CONTOURS layer instead of MAJOR/MINOR.",
                    )
                )

        return issues

    def validate_or_raise(self, feature: Feature) -> list[ValidationIssue]:
        """
        Severity x strict outcome matrix:

            WARNING, strict=True  -> feature exported, warning recorded
            WARNING, strict=False -> feature exported, warning recorded
            ERROR,   strict=True  -> raises DXFValidationError
            ERROR,   strict=False -> feature skipped, warning recorded

        `strict` only ever changes what happens to ERROR-severity issues.
        Convenience for standalone use; `DXFExporter` inlines this
        same logic itself so it can count skips correctly -- see
        `exporter.py`.
        """
        issues = self.validate(feature)
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        if errors:
            raise DXFValidationError(
                f"Feature {feature.feature_id} failed DXF validation: "
                f"{'; '.join(f'[{i.code}] {i.message}' for i in errors)}"
            )
        return issues


__all__ = ["ValidationSeverity", "ValidationIssue", "DXFValidator"]
