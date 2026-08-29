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

from .exceptions import ColumnMappingError
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
        # PR21 remediation (COLUMN-MAPPER-001): previously,
        # `mapped[target] = values` silently overwrote an earlier
        # entry whenever two different source columns normalized to
        # the same target name (e.g. "X" and "Easting" both -> "x"),
        # discarding one column's data with no warning -- confirmed
        # directly with real code before this change. TopoCore's own
        # "fail fast, no silent data loss" philosophy means an
        # ambiguous file should be explicitly rejected rather than
        # having the mapper silently pick a winner.
        #
        # Collisions are detected in a first pass, before any part of
        # `mapped` is populated -- if the file is ambiguous, nothing
        # is returned at all, not a partially-built result.
        targets: dict[str, list[str]] = {}

        for name in batch.columns:
            key = name.lower().replace(" ", "").replace("_", "").replace("-", "")
            target = cls.DEFAULT_MAPPING.get(key, key)
            targets.setdefault(target, []).append(name)

        collisions = {target: sources for target, sources in targets.items() if len(sources) > 1}

        if collisions:
            details = "; ".join(f"{sources!r} all map to {target!r}" for target, sources in sorted(collisions.items()))
            raise ColumnMappingError(f"Column normalization collision: {details}.")

        mapped = {}

        for name, values in batch.columns.items():
            key = name.lower().replace(" ", "").replace("_", "").replace("-", "")

            mapped[cls.DEFAULT_MAPPING.get(key, key)] = values

        return ASCIIRecordBatch(mapped)
