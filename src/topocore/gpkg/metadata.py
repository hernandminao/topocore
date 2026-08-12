"""
topocore.gpkg.metadata
==========================

Table naming (``<category>_<geometry family>``) and construction of
``gpkg_contents``/``gpkg_geometry_columns`` rows.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.features.models import FeatureCategory
from topocore.gpkg.geometry import GeometryFamily


def feature_table_name(category: FeatureCategory, family: GeometryFamily) -> str:
    """
    ``<category>_<geometry family>``, e.g. ``"building_polygon"``.
    Both inputs come from closed vocabularies (8 categories x 4
    families), so this is always a safe SQL identifier -- never
    built from unsanitized external input.
    """
    return f"{category.value}_{family.value}"


@dataclass(slots=True)
class TableBounds:
    """Mutable accumulator for one feature table's 2D bounding box."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def extend(self, min_x: float, min_y: float, max_x: float, max_y: float) -> None:
        self.min_x = min(self.min_x, min_x)
        self.min_y = min(self.min_y, min_y)
        self.max_x = max(self.max_x, max_x)
        self.max_y = max(self.max_y, max_y)

    @classmethod
    def from_first(cls, min_x: float, min_y: float, max_x: float, max_y: float) -> TableBounds:
        return cls(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


def contents_row(
    table_name: str,
    identifier: str,
    bounds: TableBounds,
    srs_id: int,
) -> tuple[str, str, str, str, float, float, float, float, int]:
    """Values for one gpkg_contents row, in column order (table_name, data_type, identifier, description, min_x, min_y, max_x, max_y, srs_id)."""
    return (
        table_name,
        "features",
        identifier,
        "",
        bounds.min_x,
        bounds.min_y,
        bounds.max_x,
        bounds.max_y,
        srs_id,
    )


def geometry_columns_row(
    table_name: str,
    geometry_type_name: str,
    srs_id: int,
) -> tuple[str, str, str, int, int, int]:
    """Values for one gpkg_geometry_columns row (table_name, column_name, geometry_type_name, srs_id, z, m). z=1 always -- TopoCore never flattens."""
    return (table_name, "geom", geometry_type_name, srs_id, 1, 0)


__all__ = ["TableBounds", "contents_row", "feature_table_name", "geometry_columns_row"]
