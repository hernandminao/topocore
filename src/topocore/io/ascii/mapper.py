"""
topocore.io.ascii.mapper
========================

Column mapping utilities for ASCII point cloud files.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .records import ASCIIRecordBatch


class ColumnMapper:
    """
    Maps external column names to TopoCore names.
    """

    DEFAULT_MAPPING = {
        "easting": "x",
        "east": "x",
        "longitude": "x",
        "lon": "x",
        "x": "x",
        "northing": "y",
        "north": "y",
        "latitude": "y",
        "lat": "y",
        "y": "y",
        "elevation": "z",
        "height": "z",
        "rl": "z",
        "level": "z",
        "z": "z",
        "description": "description",
        "desc": "description",
        "code": "code",
        "pointid": "id",
        "point_id": "id",
        "id": "id",
    }

    @classmethod
    def normalize(
        cls,
        batch: ASCIIRecordBatch,
    ) -> ASCIIRecordBatch:

        mapped = {}

        for name, values in batch.columns.items():
            key = name.lower().replace(" ", "").replace("_", "").replace("-", "")

            mapped[cls.DEFAULT_MAPPING.get(key, key)] = values

        return ASCIIRecordBatch(mapped)
