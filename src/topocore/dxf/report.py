from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from topocore.dxf.models import DrawingUnits
from topocore.features.models import FeatureType


@dataclass(frozen=True, slots=True)
class DXFExportReport:
    output_path: Path
    dxf_version: str
    units: DrawingUnits
    feature_count: int
    entity_count: int
    skipped_features: int
    point_count: int
    lwpolyline_count: int
    polyline3d_count: int
    face3d_count: int
    layer_count: int
    features_by_type: Mapping[FeatureType, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "dxf_version": self.dxf_version,
            "units": self.units.value,
            "feature_count": self.feature_count,
            "entity_count": self.entity_count,
            "skipped_features": self.skipped_features,
            "point_count": self.point_count,
            "lwpolyline_count": self.lwpolyline_count,
            "polyline3d_count": self.polyline3d_count,
            "face3d_count": self.face3d_count,
            "layer_count": self.layer_count,
            "features_by_type": {k.value: v for k, v in self.features_by_type.items()},
            "warnings": list(self.warnings),
            "warning_count": self.warning_count,
        }


@dataclass(slots=True)
class _ReportBuilder:
    output_path: Path = field(default_factory=Path)
    dxf_version: str = ""
    units: DrawingUnits = DrawingUnits.METERS
    feature_count: int = 0
    entity_count: int = 0
    skipped_features: int = 0
    point_count: int = 0
    lwpolyline_count: int = 0
    polyline3d_count: int = 0
    face3d_count: int = 0
    layer_count: int = 0
    features_by_type: Counter[FeatureType] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def finalize(self) -> DXFExportReport:
        return DXFExportReport(
            output_path=self.output_path,
            dxf_version=self.dxf_version,
            units=self.units,
            feature_count=self.feature_count,
            entity_count=self.entity_count,
            skipped_features=self.skipped_features,
            point_count=self.point_count,
            lwpolyline_count=self.lwpolyline_count,
            polyline3d_count=self.polyline3d_count,
            face3d_count=self.face3d_count,
            layer_count=self.layer_count,
            features_by_type=dict(self.features_by_type),
            warnings=tuple(self.warnings),
        )


__all__ = ["DXFExportReport", "_ReportBuilder"]
