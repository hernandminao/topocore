from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from topocore.dxf.exceptions import DXFGeometryError
from topocore.dxf.models import NonPlanarPolygonMode
from topocore.dxf.tolerance import DXFTolerance
from topocore.features.models import FeatureGeometry, GeometryType


class DXFRepresentation(StrEnum):
    POINT = "point"
    LWPOLYLINE = "lwpolyline"
    POLYLINE3D = "polyline3d"
    FACE3D = "3dface"


@dataclass(frozen=True, slots=True)
class MappingDecision:
    representation: DXFRepresentation
    elevation: float | None = None


def has_constant_elevation(vertices: NDArray[np.float64], tolerance: DXFTolerance) -> bool:
    z = vertices[:, 2]
    return bool(float(z.max() - z.min()) <= tolerance.z_planarity)


class GeometryMapper:
    __slots__ = ("_tolerance", "_non_planar_polygon_mode")

    def __init__(self, tolerance: DXFTolerance, *, non_planar_polygon_mode: NonPlanarPolygonMode) -> None:
        self._tolerance = tolerance
        self._non_planar_polygon_mode = non_planar_polygon_mode

    def decide(self, geometry: FeatureGeometry) -> MappingDecision:
        if geometry.geometry_type == GeometryType.POINT:
            return MappingDecision(DXFRepresentation.POINT)

        if geometry.geometry_type == GeometryType.MESH:
            return MappingDecision(DXFRepresentation.FACE3D)

        constant_z = has_constant_elevation(geometry.vertices, self._tolerance)

        if geometry.geometry_type == GeometryType.POLYLINE:
            if constant_z:
                return MappingDecision(DXFRepresentation.LWPOLYLINE, elevation=float(np.mean(geometry.vertices[:, 2])))
            return MappingDecision(DXFRepresentation.POLYLINE3D)

        if geometry.geometry_type == GeometryType.POLYGON:
            if constant_z:
                return MappingDecision(DXFRepresentation.LWPOLYLINE, elevation=float(np.mean(geometry.vertices[:, 2])))
            if self._non_planar_polygon_mode == NonPlanarPolygonMode.ERROR:
                raise DXFGeometryError(
                    "Non-planar POLYGON cannot be represented as a closed LWPOLYLINE "
                    "without losing Z variation. Set DXFExportOptions."
                    "non_planar_polygon_mode=NonPlanarPolygonMode.POLYLINE3D to preserve "
                    "XYZ as a closed 3D polyline instead."
                )
            return MappingDecision(DXFRepresentation.POLYLINE3D)

        raise DXFGeometryError(f"No DXF mapping defined for {geometry.geometry_type.value}.")


__all__ = ["DXFRepresentation", "MappingDecision", "has_constant_elevation", "GeometryMapper"]
