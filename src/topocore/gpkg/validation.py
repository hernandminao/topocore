"""
topocore.gpkg.validation
============================

Pre-write validation for a single Feature, mirroring
``dxf.validation``'s role for PR16: catches problems before they
become a malformed row in the database.

Found and fixed during this project's own GeoPackage documentation
audit: this validator's own checks used to be a strict subset of
what `gpkg.geometry._to_shapely()` (called later, inside the export
transaction) actually requires -- missing a finite-coordinate check
(``GPKG006``) and a degenerate-MESH-triangle check (``GPKG007``),
both now added here, mirroring `_to_shapely()`'s own checks exactly.
A `Feature` that only `_to_shapely()` rejected used to reach
`GeoPackageExporter._write()` inside the write transaction, where
any `GPKGGeometryError` is treated as fatal to the WHOLE export
regardless of `GPKGExportOptions.strict` -- silently violating
`strict=False`'s own documented contract ("that feature is skipped
and recorded in the report instead"), since the isolate-and-skip
logic only runs in `GeoPackageExporter.export()`'s own earlier
per-feature validation loop, before `_write()` is ever called. With
both checks added here, this validator's own coverage matches
`_to_shapely()`'s exactly, so nothing reaches the transaction that
wasn't already accepted here first.

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

        vertices = feature.geometry.vertices

        if not np.isfinite(vertices).all():
            issues.append(
                GPKGValidationIssue(
                    code="GPKG006",
                    feature_id=feature.feature_id,
                    severity=GPKGValidationSeverity.ERROR,
                    message="Geometry vertices contain NaN or infinite coordinates.",
                )
            )

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
            elif np.isfinite(vertices).all():
                # Degenerate-face check mirrors `gpkg.geometry._to_shapely()`'s
                # own check exactly (same cross-product test), so a
                # feature that would fail there is already caught here,
                # before the export transaction -- see GPKGValidationIssue
                # GPKG007's own account in this project's own
                # documentation for why this matters for `strict=False`.
                # Skipped when vertices are already non-finite (GPKG006
                # above already reports that; the cross product itself
                # would just be NaN, not a meaningful degeneracy signal).
                for face in faces:
                    tri = vertices[face]
                    edge_a = tri[1] - tri[0]
                    edge_b = tri[2] - tri[0]
                    cross = np.cross(edge_a, edge_b)

                    if float(np.dot(cross, cross)) <= 0.0:
                        issues.append(
                            GPKGValidationIssue(
                                code="GPKG007",
                                feature_id=feature.feature_id,
                                severity=GPKGValidationSeverity.ERROR,
                                message="MESH contains a degenerate triangular face.",
                            )
                        )
                        break

        return tuple(issues)


__all__ = ["GPKGValidationIssue", "GPKGValidationSeverity", "GPKGValidator"]
