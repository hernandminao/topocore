"""
topocore.gpkg.schema
========================

DDL for the mandatory GeoPackage system tables and for the dynamic
feature tables TopoCore creates (one per ``<category>_<geometry
family>`` combination that actually has features -- see
``gpkg.metadata`` for how table names are derived).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

CREATE_GPKG_SPATIAL_REF_SYS = """
CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (
    srs_name TEXT NOT NULL,
    srs_id INTEGER NOT NULL PRIMARY KEY,
    organization TEXT NOT NULL,
    organization_coordsys_id INTEGER NOT NULL,
    definition TEXT NOT NULL,
    description TEXT
);
"""

CREATE_GPKG_CONTENTS = """
CREATE TABLE IF NOT EXISTS gpkg_contents (
    table_name TEXT NOT NULL PRIMARY KEY,
    data_type TEXT NOT NULL,
    identifier TEXT UNIQUE,
    description TEXT DEFAULT '',
    last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    min_x DOUBLE,
    min_y DOUBLE,
    max_x DOUBLE,
    max_y DOUBLE,
    srs_id INTEGER,
    CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
);
"""

CREATE_GPKG_GEOMETRY_COLUMNS = """
CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    geometry_type_name TEXT NOT NULL,
    srs_id INTEGER NOT NULL,
    z TINYINT NOT NULL,
    m TINYINT NOT NULL,
    CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
    CONSTRAINT uk_gc_table_name UNIQUE (table_name),
    CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
    CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
);
"""

CREATE_GPKG_EXTENSIONS = """
CREATE TABLE IF NOT EXISTS gpkg_extensions (
    table_name TEXT,
    column_name TEXT,
    extension_name TEXT NOT NULL,
    definition TEXT NOT NULL,
    scope TEXT NOT NULL,
    CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name)
);
"""

SYSTEM_TABLE_DDL: tuple[str, ...] = (
    CREATE_GPKG_SPATIAL_REF_SYS,
    CREATE_GPKG_CONTENTS,
    CREATE_GPKG_GEOMETRY_COLUMNS,
    CREATE_GPKG_EXTENSIONS,
)


def create_feature_table_sql(table_name: str, geometry_type_name: str) -> str:
    """
    DDL for one feature table. `table_name` must already be a
    validated, internally-generated identifier (see
    `gpkg.metadata.feature_table_name`) -- never built from
    unsanitized external input.
    """
    return f"""
    CREATE TABLE "{table_name}" (
        fid INTEGER PRIMARY KEY AUTOINCREMENT,
        geom {geometry_type_name},
        feature_id INTEGER,
        feature_type TEXT NOT NULL,
        category TEXT NOT NULL,
        survey_code TEXT,
        survey_name TEXT,
        cad_layer TEXT,
        confidence REAL,
        producer TEXT,
        producer_version TEXT,
        attributes_json TEXT
    );
    """


def create_rtree_table_sql(table_name: str) -> str:
    return f'CREATE VIRTUAL TABLE "rtree_{table_name}_geom" USING rtree(id, minx, maxx, miny, maxy);'


__all__ = [
    "SYSTEM_TABLE_DDL",
    "create_feature_table_sql",
    "create_rtree_table_sql",
]
