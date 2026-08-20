"""
topocore.gpkg.validation
============================

Pre-write validation for a single Feature, mirroring
``dxf.validation``'s role for PR16: catches problems before they
become a malformed row in the database.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from topocore.features.models import Feature, GeometryType


class GPKGValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class GPKGValidationIssue:
    code: str
    feature_id: int | None
    severity: GPKGValidationSeverity
    message: str


class GPKGValidator:
    """Stateless; `validate()` never mutates the Feature."""

    def validate(self, feature: Feature) -> tuple[GPKGValidationIssue, ...]:
        issues: list[GPKGValidationIssue] = []

        if feature.geometry.geometry_type == GeometryType.POLYGON and not feature.geometry.closed:
            issues.append(
                GPKGValidationIssue(
                    code="GPKG001",
                    feature_id=feature.feature_id,
                    severity=GPKGValidationSeverity.ERROR,
                    message="POLYGON geometry must be closed to produce a valid GeoPackage ring.",
                )
            )

        if feature.geometry.geometry_type == GeometryType.MESH:
            faces = feature.geometry.faces
            vertices = feature.geometry.vertices

            if faces is None or faces.size == 0:
                issues.append(
                    GPKGValidationIssue(
                        code="GPKG002",
                        feature_id=feature.feature_id,
                        severity=GPKGValidationSeverity.ERROR,
                        message=("MESH geometry has no faces to triangulate into a MultiPolygon."),
                    )
                )
            elif faces.ndim != 2 or faces.shape[1] != 3:
                issues.append(
                    GPKGValidationIssue(
                        code="GPKG003",
                        feature_id=feature.feature_id,
                        severity=GPKGValidationSeverity.ERROR,
                        message="MESH faces must have shape (n, 3).",
                    )
                )
            elif not np.issubdtype(faces.dtype, np.integer):
                issues.append(
                    GPKGValidationIssue(
                        code="GPKG004",
                        feature_id=feature.feature_id,
                        severity=GPKGValidationSeverity.ERROR,
                        message="MESH face indices must be integers.",
                    )
                )
            elif np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
                issues.append(
                    GPKGValidationIssue(
                        code="GPKG005",
                        feature_id=feature.feature_id,
                        severity=GPKGValidationSeverity.ERROR,
                        message="MESH contains out-of-range face indices.",
                    )
                )

        return tuple(issues)


__all__ = ["GPKGValidationIssue", "GPKGValidationSeverity", "GPKGValidator"]
