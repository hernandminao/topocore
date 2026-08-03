"""
topocore.analysis.volume.tin_volume
====================================

TIN-based volume computation.

Computes the volume between a triangulated surface and a reference
datum plane using triangular prism integration.

For each triangle:

    V = A_xy * ((z1 + z2 + z3) / 3)

where elevations are measured relative to the datum plane.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.analysis.exceptions import VolumeError
from topocore.analysis.protocols import TriangulatedSurface
from topocore.analysis.types import VolumeResult
from topocore.geometry.point3d import Point3D


class TINVolume:
    """
    Computes volume below a TIN surface relative to a datum.

    Parameters
    ----------
    datum
        Reference elevation plane.
    """

    __slots__ = ("_datum",)

    def __init__(
        self,
        datum: float = 0.0,
    ) -> None:

        if not math.isfinite(datum):
            raise VolumeError("Datum elevation must be finite.")

        self._datum = float(datum)

    @property
    def datum(self) -> float:
        """Reference datum elevation."""
        return self._datum

    def compute(
        self,
        tin: TriangulatedSurface,
    ) -> VolumeResult:
        """
        Compute volume between TIN and datum plane.
        """

        if tin.triangle_count <= 0:
            raise VolumeError("TIN contains no triangles.")

        cut_volume = 0.0
        fill_volume = 0.0

        for index in range(tin.triangle_count):
            p1, p2, p3 = tin.triangle_vertices(index)

            self._validate_vertex(p1, index)
            self._validate_vertex(p2, index)
            self._validate_vertex(p3, index)

            area = self._triangle_area_xy(
                p1.x,
                p1.y,
                p2.x,
                p2.y,
                p3.x,
                p3.y,
            )

            if area <= 0.0:
                continue

            z1 = p1.z - self._datum
            z2 = p2.z - self._datum
            z3 = p3.z - self._datum

            mean_height = (z1 + z2 + z3) / 3.0

            volume = area * mean_height

            if volume >= 0.0:
                cut_volume += volume
            else:
                fill_volume -= volume

        return VolumeResult(
            cut_volume=cut_volume,
            fill_volume=fill_volume,
            net_volume=cut_volume - fill_volume,
            method="tin_volume",
        )

    @staticmethod
    def _triangle_area_xy(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
    ) -> float:
        """
        Compute projected triangle area.
        """

        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) * 0.5)

    @staticmethod
    def _validate_vertex(
        vertex: Point3D,
        index: int,
    ) -> None:
        """
        Validate triangle vertex.
        """

        if not (math.isfinite(vertex.x) and math.isfinite(vertex.y) and math.isfinite(vertex.z)):
            raise VolumeError(f"Triangle {index} contains invalid coordinates.")

    def __call__(
        self,
        tin: TriangulatedSurface,
    ) -> VolumeResult:

        return self.compute(tin)


__all__ = [
    "TINVolume",
]
