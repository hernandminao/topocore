"""
topocore.gpkg.exporter
==========================

Orchestrates a `FeatureCollection` -> GeoPackage export: system
tables, one feature table per (category, geometry family) that
actually has features, R*Tree spatial index population (see module
docstring note on trigger scope), all inside a single transaction,
written to a temp file and atomically renamed into place so a failed
export never leaves a corrupt ``.gpkg`` at the target path.

R*Tree scope note
------------------
The R*Tree virtual table is created and fully populated at export
time -- spatial queries against the finished file are correctly
accelerated. The trigger-based auto-maintenance the OGC spec
describes for keeping the index in sync after later edits is NOT
implemented: those triggers depend on spatial SQL functions
(``ST_MinX``, etc.) that don't exist in plain ``sqlite3`` without a
loaded spatial extension (exactly the GDAL dependency this
implementation avoids). TopoCore only ever produces complete,
finished GeoPackages in one pass -- it doesn't edit existing ones --
so this is a scoped, documented limitation, not a silent gap.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

from topocore.features.models import Feature, FeatureCategory, FeatureCollection
from topocore.gpkg.config import (
    GPKG_APPLICATION_ID,
    GPKG_USER_VERSION,
    GPKGExportOptions,
)
from topocore.gpkg.exceptions import (
    GPKGExportError,
    GPKGGeometryError,
    GPKGValidationError,
)
from topocore.gpkg.geometry import (
    GEOMETRY_TYPE_NAME,
    GeometryFamily,
    build_gpb,
    geometry_bounds_2d,
    geometry_family,
)
from topocore.gpkg.metadata import (
    TableBounds,
    contents_row,
    feature_table_name,
    geometry_columns_row,
)
from topocore.gpkg.report import GPKGExportReport, _ReportBuilder
from topocore.gpkg.schema import (
    SYSTEM_TABLE_DDL,
    create_feature_table_sql,
    create_rtree_table_sql,
)
from topocore.gpkg.spatial_ref import ResolvedSRS, resolve_srs
from topocore.gpkg.validation import GPKGValidationSeverity, GPKGValidator

_PROMOTED_ATTRIBUTE_KEYS = ("survey_code", "survey_name", "cad_layer")

_INSERT_SRS_SQL = (
    "INSERT OR IGNORE INTO gpkg_spatial_ref_sys "
    "(srs_id, srs_name, organization, organization_coordsys_id, definition, description) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)

#: One row for a feature table INSERT (see create_feature_table_sql's column order).
_FeatureRow = tuple[
    int,  # fid
    bytes,  # geom (GPB blob)
    int | None,  # feature_id
    str,  # feature_type
    str,  # category
    str | None,  # survey_code
    str | None,  # survey_name
    str | None,  # cad_layer
    float,  # confidence
    str | None,  # producer
    str | None,  # producer_version
    str | None,  # attributes_json
]

#: One row for an R*Tree virtual table INSERT (id, minx, maxx, miny, maxy).
_RTreeRow = tuple[int, float, float, float, float]

#: Mandatory GeoPackage-spec placeholder rows, always present regardless of the requested EPSG.
_BOILERPLATE_SRS_ROWS: tuple[tuple[int, str, str, int, str, str], ...] = (
    (
        -1,
        "Undefined cartesian SRS",
        "NONE",
        -1,
        "undefined",
        "undefined Cartesian coordinate reference system",
    ),
    (
        0,
        "Undefined geographic SRS",
        "NONE",
        0,
        "undefined",
        "undefined geographic coordinate reference system",
    ),
)
_WGS84_EPSG = 4326


class GeoPackageExporter:
    __slots__ = ("_options", "_validator")

    def __init__(self, options: GPKGExportOptions) -> None:
        self._options = options
        self._validator = GPKGValidator()

    def export(self, collection: FeatureCollection, path: str | Path) -> GPKGExportReport:
        resolved = resolve_srs(self._options.epsg)
        final_path = Path(path)
        report = _ReportBuilder()

        groups: dict[tuple[FeatureCategory, GeometryFamily], list[Feature]] = {}

        for feature in collection:
            report.feature_count += 1
            issues = self._validator.validate(feature)
            errors = [i for i in issues if i.severity == GPKGValidationSeverity.ERROR]

            if errors:
                if self._options.strict:
                    raise GPKGValidationError(
                        f"Feature {feature.feature_id} failed GeoPackage validation: "
                        f"{'; '.join(f'[{i.code}] {i.message}' for i in errors)}"
                    )
                report.skipped_count += 1
                report.warnings.append(f"feature {feature.feature_id} skipped: {errors[0].message}")
                continue

            try:
                family = geometry_family(feature.geometry.geometry_type)
            except GPKGGeometryError as exc:
                if self._options.strict:
                    raise GPKGValidationError(
                        f"Feature {feature.feature_id} failed GeoPackage validation: {exc}"
                    ) from exc
                report.skipped_count += 1
                report.warnings.append(f"feature {feature.feature_id} skipped: {exc}")
                continue

            key = (feature.category, family)
            groups.setdefault(key, []).append(feature)

        # Validation is fully done at this point -- only now do we
        # touch disk at all. `dir=` MUST be the same directory as
        # final_path, or os.replace() below stops being atomic (it
        # only guarantees atomicity within one filesystem/volume).
        fd, temp_name = tempfile.mkstemp(suffix=".tmp", prefix=f"{final_path.name}.", dir=final_path.parent)
        os.close(fd)
        temp_path = Path(temp_name)

        try:
            self._write(temp_path, resolved, groups, report)
            os.replace(temp_path, final_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

        return report.finalize(final_path, resolved.srs_id)

    def _write(
        self,
        temp_path: Path,
        resolved: ResolvedSRS,
        groups: dict[tuple[FeatureCategory, GeometryFamily], list[Feature]],
        report: _ReportBuilder,
    ) -> None:
        conn = sqlite3.connect(temp_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(f"PRAGMA application_id = {GPKG_APPLICATION_ID};")
            conn.execute(f"PRAGMA user_version = {GPKG_USER_VERSION};")

            for ddl in SYSTEM_TABLE_DDL:
                conn.execute(ddl)

            self._write_srs_rows(conn, resolved)

            for (category, family), features in groups.items():
                self._write_feature_table(conn, category, family, features, resolved.srs_id, report)

            conn.commit()
        except (sqlite3.Error, GPKGGeometryError) as exc:
            conn.rollback()
            raise GPKGExportError(f"GeoPackage export failed: {exc}") from exc
        finally:
            conn.close()

    def _write_srs_rows(self, conn: sqlite3.Connection, resolved: ResolvedSRS) -> None:
        for row in _BOILERPLATE_SRS_ROWS:
            conn.execute(
                _INSERT_SRS_SQL,
                row,
            )

        if resolved.srs_id != _WGS84_EPSG:
            try:
                wgs84 = resolve_srs(_WGS84_EPSG)
                conn.execute(
                    _INSERT_SRS_SQL,
                    (
                        wgs84.srs_id,
                        wgs84.srs_name,
                        wgs84.organization,
                        wgs84.organization_coordsys_id,
                        wgs84.definition,
                        wgs84.description,
                    ),
                )
            except GPKGExportError:
                pass  # WGS84 is a courtesy convenience row, never required for the export to succeed

        conn.execute(
            _INSERT_SRS_SQL,
            (
                resolved.srs_id,
                resolved.srs_name,
                resolved.organization,
                resolved.organization_coordsys_id,
                resolved.definition,
                resolved.description,
            ),
        )

    def _write_feature_table(
        self,
        conn: sqlite3.Connection,
        category: FeatureCategory,
        family: GeometryFamily,
        features: list[Feature],
        srs_id: int,
        report: _ReportBuilder,
    ) -> None:
        if not features:
            raise ValueError("_write_feature_table() requires a non-empty feature list.")

        table_name = feature_table_name(category, family)
        geometry_type_name = GEOMETRY_TYPE_NAME[family]

        conn.execute(create_feature_table_sql(table_name, geometry_type_name))
        conn.execute(create_rtree_table_sql(table_name))
        conn.execute(
            "INSERT INTO gpkg_extensions (table_name, column_name, extension_name, definition, scope) "
            "VALUES (?, 'geom', 'gpkg_rtree_index', 'http://www.geopackage.org/spec/#extension_rtree', 'write-only')",
            (table_name,),
        )

        # Fresh, empty table for this group -> AUTOINCREMENT hands
        # out exactly 1..N in insertion order (verified against real
        # sqlite3 behavior). Assigning fid explicitly lets both the
        # feature rows and the R*Tree rows go through executemany()
        # in one batch each, instead of one execute() call per row --
        # executemany() doesn't expose cursor.lastrowid per row, so
        # we can't rely on it for the R*Tree insert otherwise.
        feature_rows: list[_FeatureRow] = []
        rtree_rows: list[_RTreeRow] = []
        bounds: TableBounds | None = None

        for fid, feature in enumerate(features, start=1):
            gpb = build_gpb(feature.geometry, srs_id)
            min_x, min_y, max_x, max_y = geometry_bounds_2d(feature.geometry)

            survey_code, survey_name, cad_layer, attributes_json = self._split_attributes(feature)
            producer = feature.metadata.detector if feature.metadata else None
            producer_version = feature.metadata.version if feature.metadata else None

            feature_rows.append(
                (
                    fid,
                    gpb,
                    feature.feature_id,
                    feature.feature_type.value,
                    feature.category.value,
                    survey_code,
                    survey_name,
                    cad_layer,
                    feature.confidence,
                    producer,
                    producer_version,
                    attributes_json,
                )
            )
            rtree_rows.append((fid, min_x, max_x, min_y, max_y))

            if bounds is None:
                bounds = TableBounds.from_first(min_x, min_y, max_x, max_y)
            else:
                bounds.extend(min_x, min_y, max_x, max_y)

            report.record_written(table_name)

        conn.executemany(
            f'INSERT INTO "{table_name}" '
            "(fid, geom, feature_id, feature_type, category, survey_code, survey_name, cad_layer, "
            "confidence, producer, producer_version, attributes_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            feature_rows,
        )
        conn.executemany(
            f'INSERT INTO "rtree_{table_name}_geom" (id, minx, maxx, miny, maxy) VALUES (?, ?, ?, ?, ?)',
            rtree_rows,
        )

        assert bounds is not None  # `features` is non-empty (checked above), so the loop ran at least once

        conn.execute(
            "INSERT INTO gpkg_contents "
            "(table_name, data_type, identifier, description, min_x, min_y, max_x, max_y, srs_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            contents_row(table_name, table_name, bounds, srs_id),
        )
        conn.execute(
            "INSERT INTO gpkg_geometry_columns (table_name, column_name, geometry_type_name, srs_id, z, m) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            geometry_columns_row(table_name, geometry_type_name, srs_id),
        )

    @staticmethod
    def _split_attributes(
        feature: Feature,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        attrs = feature.attributes
        survey_code = attrs.get("survey_code")
        survey_name = attrs.get("survey_name")
        cad_layer = attrs.get("cad_layer")
        rest = {k: v for k, v in attrs.items() if k not in _PROMOTED_ATTRIBUTE_KEYS}
        attributes_json = json.dumps(rest, ensure_ascii=False, separators=(",", ":")) if rest else None
        return survey_code, survey_name, cad_layer, attributes_json


__all__ = ["GeoPackageExporter"]
