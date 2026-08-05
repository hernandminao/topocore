from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from topocore.dxf.constants import DEFAULT_DXF_VERSION, DEFAULT_INDEX_CONTOUR_EVERY
from topocore.dxf.tolerance import DXFTolerance


class DrawingUnits(StrEnum):
    METERS = "meters"
    MILLIMETERS = "millimeters"
    FEET = "feet"


class NonPlanarPolygonMode(StrEnum):
    POLYLINE3D = "polyline3d"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LayerStyle:
    name: str
    color: int
    linetype: str = "CONTINUOUS"
    lineweight: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.color <= 255:
            raise ValueError(f"color must be a valid ACI (1-255); got {self.color}.")


@dataclass(frozen=True, slots=True)
class DXFExportOptions:
    dxf_version: str = DEFAULT_DXF_VERSION
    units: DrawingUnits = DrawingUnits.METERS
    non_planar_polygon_mode: NonPlanarPolygonMode = NonPlanarPolygonMode.POLYLINE3D
    tolerance: DXFTolerance = field(default_factory=DXFTolerance)
    strict: bool = True
    index_contour_every: int = DEFAULT_INDEX_CONTOUR_EVERY

    def __post_init__(self) -> None:
        if self.index_contour_every < 1:
            raise ValueError(f"index_contour_every must be >= 1; got {self.index_contour_every}.")


@dataclass(frozen=True, slots=True)
class ExportContext:
    crs: str | None = None
    options: DXFExportOptions = field(default_factory=DXFExportOptions)


__all__ = ["DrawingUnits", "NonPlanarPolygonMode", "LayerStyle", "DXFExportOptions", "ExportContext"]
