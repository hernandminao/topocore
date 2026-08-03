"""
topocore.analysis.statistics.area
==================================

Area statistics.

Computes projected (planimetric) and true 3D surface
areas from a triangulated terrain surface.

The projected area represents the horizontal footprint,
while the surface area accounts for terrain inclination.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from typing import Final

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.protocols import TriangulatedSurface
from topocore.analysis.types import AreaStats
from topocore.geometry.point3d import Point3D

_HALF: Final[float] = 0.5


class AreaStatistics:
    """
    Computes area statistics from a TIN surface.
    """

    __slots__ = ()

    @staticmethod
    def compute(
        tin: TriangulatedSurface,
    ) -> AreaStats:
        """
        Compute projected and surface areas of a TIN.

        Parameters
        ----------
        tin
            Triangulated surface.

        Returns
        -------
        AreaStats
            Projected area, surface area, and triangle count.

        Raises
        ------
        StatisticsError
            If the TIN contains no triangles.
        """
        triangle_count = tin.triangle_count

        if triangle_count == 0:
            raise StatisticsError("TIN has no triangles.")

        projected_area = 0.0
        surface_area = 0.0

        for index in range(triangle_count):
            p1, p2, p3 = tin.triangle_vertices(index)

            projected_area += AreaStatistics.projected_triangle_area(
                p1,
                p2,
                p3,
            )

            surface_area += AreaStatistics.triangle_area(
                p1,
                p2,
                p3,
            )

        return AreaStats(
            total_area=projected_area,
            projected_area=projected_area,
            surface_area=surface_area,
            count=triangle_count,
        )

    @staticmethod
    def triangle_area(
        p1: Point3D,
        p2: Point3D,
        p3: Point3D,
    ) -> float:
        """
        Compute the 3D surface area of a triangle.

        Uses the cross-product formulation:

            A = 0.5 * ||(P2-P1) × (P3-P1)||

        Parameters
        ----------
        p1, p2, p3
            Triangle vertices.

        Returns
        -------
        float
            Triangle surface area.
        """
        v1_x = p2.x - p1.x
        v1_y = p2.y - p1.y
        v1_z = p2.z - p1.z

        v2_x = p3.x - p1.x
        v2_y = p3.y - p1.y
        v2_z = p3.z - p1.z

        cross_x = v1_y * v2_z - v1_z * v2_y
        cross_y = v1_z * v2_x - v1_x * v2_z
        cross_z = v1_x * v2_y - v1_y * v2_x

        return math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z) * _HALF

    @staticmethod
    def projected_triangle_area(
        p1: Point3D,
        p2: Point3D,
        p3: Point3D,
    ) -> float:
        """
        Compute projected XY area of a triangle.

        Uses the shoelace formula.

        Parameters
        ----------
        p1, p2, p3
            Triangle vertices.

        Returns
        -------
        float
            Horizontal projected area.
        """
        return abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) * _HALF)

    def __call__(
        self,
        tin: TriangulatedSurface,
    ) -> AreaStats:
        """
        Execute area statistics computation.
        """
        return self.compute(tin)


__all__ = [
    "AreaStatistics",
]
