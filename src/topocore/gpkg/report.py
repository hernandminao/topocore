"""
topocore.gpkg.report
========================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GPKGExportReport:
    output_path: Path
    srid: int
    feature_count: int
    written_count: int
    skipped_count: int
    table_count: int
    features_by_table: Mapping[str, int]
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class _ReportBuilder:
    feature_count: int = 0
    written_count: int = 0
    skipped_count: int = 0
    features_by_table: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def record_written(self, table_name: str) -> None:
        self.written_count += 1
        self.features_by_table[table_name] = self.features_by_table.get(table_name, 0) + 1

    def finalize(self, output_path: Path, srid: int) -> GPKGExportReport:
        return GPKGExportReport(
            output_path=output_path,
            srid=srid,
            feature_count=self.feature_count,
            written_count=self.written_count,
            skipped_count=self.skipped_count,
            table_count=len(self.features_by_table),
            features_by_table=dict(self.features_by_table),
            warnings=tuple(self.warnings),
        )


__all__ = ["GPKGExportReport"]
