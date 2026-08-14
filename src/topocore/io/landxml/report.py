"""
topocore.io.landxml.report
============================

Read/write reports for the TopoCore LandXML IO subsystem.

Mirrors the ``topocore.dxf.report`` pattern: an immutable, public
report returned to the caller, plus a small mutable builder used
internally while parsing/serializing.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LandXMLReadReport:
    input_path: Path
    surface_count: int
    point_group_count: int
    alignment_count: int
    triangle_count: int
    point_count: int
    warnings: tuple[str, ...] = ()

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "surface_count": self.surface_count,
            "point_group_count": self.point_group_count,
            "alignment_count": self.alignment_count,
            "triangle_count": self.triangle_count,
            "point_count": self.point_count,
            "warnings": list(self.warnings),
            "warning_count": self.warning_count,
        }


@dataclass(slots=True)
class _ReadReportBuilder:
    input_path: Path = field(default_factory=Path)
    surface_count: int = 0
    point_group_count: int = 0
    alignment_count: int = 0
    triangle_count: int = 0
    point_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def build(self) -> LandXMLReadReport:
        return LandXMLReadReport(
            input_path=self.input_path,
            surface_count=self.surface_count,
            point_group_count=self.point_group_count,
            alignment_count=self.alignment_count,
            triangle_count=self.triangle_count,
            point_count=self.point_count,
            warnings=tuple(self.warnings),
        )


@dataclass(frozen=True, slots=True)
class LandXMLWriteReport:
    output_path: Path
    surface_count: int
    point_group_count: int
    alignment_count: int
    triangle_count: int
    point_count: int
    warnings: tuple[str, ...] = ()

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "surface_count": self.surface_count,
            "point_group_count": self.point_group_count,
            "alignment_count": self.alignment_count,
            "triangle_count": self.triangle_count,
            "point_count": self.point_count,
            "warnings": list(self.warnings),
            "warning_count": self.warning_count,
        }


@dataclass(slots=True)
class _WriteReportBuilder:
    output_path: Path = field(default_factory=Path)
    surface_count: int = 0
    point_group_count: int = 0
    alignment_count: int = 0
    triangle_count: int = 0
    point_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def build(self) -> LandXMLWriteReport:
        return LandXMLWriteReport(
            output_path=self.output_path,
            surface_count=self.surface_count,
            point_group_count=self.point_group_count,
            alignment_count=self.alignment_count,
            triangle_count=self.triangle_count,
            point_count=self.point_count,
            warnings=tuple(self.warnings),
        )


__all__ = [
    "LandXMLReadReport",
    "LandXMLWriteReport",
]
