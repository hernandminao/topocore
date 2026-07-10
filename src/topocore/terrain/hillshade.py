"""
topocore.terrain.hillshade
============================

Hillshade computation over a Triangulated Irregular Network (TIN).

Implements the standard Lambertian illumination model used by GIS
tools such as GDAL's ``gdaldem hillshade``: each facet's illumination
depends on its slope and aspect relative to a simulated sun position
(azimuth and altitude). As with ``slope.py`` and ``aspect.py``, this
is computed per triangle directly from the facet's plane, at the
resolution of the terrain model itself.

Output is scaled to the conventional 8-bit grayscale range [0, 255],
matching common raster hillshade output.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from topocore.terrain._geometry import aspect_radians, oriented_normal, slope_radians
from topocore.terrain.exceptions import TerrainValidationError
from topocore.terrain.models import Triangle
from topocore.terrain.tin import TIN

#: Standard GDAL/ArcGIS default sun position (compass degrees, NW).
DEFAULT_AZIMUTH = 315.0

#: Standard GDAL/ArcGIS default sun altitude above the horizon.
DEFAULT_ALTITUDE = 45.0

MIN_AZIMUTH = 0.0
MAX_AZIMUTH = 360.0

MIN_ALTITUDE = 0.0
MAX_ALTITUDE = 90.0


def _validate_azimuth(azimuth: float) -> None:
    if not MIN_AZIMUTH <= azimuth <= MAX_AZIMUTH:
        raise TerrainValidationError(
            f"Azimuth must be in [{MIN_AZIMUTH}, {MAX_AZIMUTH}] degrees, got {azimuth}."
        )


def _validate_altitude(altitude: float) -> None:
    if not MIN_ALTITUDE <= altitude <= MAX_ALTITUDE:
        raise TerrainValidationError(
            f"Altitude must be in [{MIN_ALTITUDE}, {MAX_ALTITUDE}] degrees, got {altitude}."
        )


def triangle_hillshade(
    triangle: Triangle,
    *,
    azimuth: float = DEFAULT_AZIMUTH,
    altitude: float = DEFAULT_ALTITUDE,
) -> float:
    """
    Compute the hillshade value of a single triangle.

    Parameters
    ----------
    azimuth
        Sun compass direction in degrees, clockwise from North.
    altitude
        Sun elevation above the horizon, in degrees.

    Returns
    -------
    float
        Illumination in the range [0, 255]. 0 means the facet is
        fully self-shadowed (facing away from the sun); 255 means
        the sun is perpendicular to the facet.
    """
    _validate_azimuth(azimuth)
    _validate_altitude(altitude)

    normal = oriented_normal(triangle)

    slope_rad = slope_radians(normal)
    aspect_rad = aspect_radians(normal)

    zenith_rad = math.radians(90.0 - altitude)
    azimuth_rad = math.radians(azimuth)

    cos_incidence = math.cos(zenith_rad) * math.cos(slope_rad)

    if aspect_rad is not None:
        cos_incidence += (
            math.sin(zenith_rad) * math.sin(slope_rad) * math.cos(azimuth_rad - aspect_rad)
        )

    return 255.0 * max(0.0, cos_incidence)


class HillshadeCalculator:
    """
    Computes per-triangle hillshade over a TIN.
    """

    __slots__ = (
        "_tin",
        "_azimuth",
        "_altitude",
    )

    def __init__(
        self,
        tin: TIN,
        *,
        azimuth: float = DEFAULT_AZIMUTH,
        altitude: float = DEFAULT_ALTITUDE,
    ) -> None:
        _validate_azimuth(azimuth)
        _validate_altitude(altitude)

        self._tin = tin
        self._azimuth = float(azimuth)
        self._altitude = float(altitude)

    @property
    def tin(
        self,
    ) -> TIN:
        return self._tin

    @property
    def azimuth(
        self,
    ) -> float:
        return self._azimuth

    @property
    def altitude(
        self,
    ) -> float:
        return self._altitude

    def _triangle_at(
        self,
        index: int,
    ) -> Triangle:
        simplex = self._tin.simplices[index]
        vertices = self._tin.vertices

        return Triangle(
            vertices[int(simplex[0])],
            vertices[int(simplex[1])],
            vertices[int(simplex[2])],
        )

    def at(
        self,
        index: int,
    ) -> float:
        """
        Hillshade of a single TIN triangle, by index.
        """
        return triangle_hillshade(
            self._triangle_at(index),
            azimuth=self._azimuth,
            altitude=self._altitude,
        )

    def compute(
        self,
    ) -> NDArray[np.float64]:
        """
        Hillshade for every triangle in the TIN.

        Returns
        -------
        ndarray
            One value per triangle, in [0, 255], in the same order
            as ``tin.simplices``.
        """
        simplices = self._tin.simplices

        values = np.empty(
            simplices.shape[0],
            dtype=np.float64,
        )

        for i in range(simplices.shape[0]):
            values[i] = triangle_hillshade(
                self._triangle_at(i),
                azimuth=self._azimuth,
                altitude=self._altitude,
            )

        return values

    def __call__(
        self,
    ) -> NDArray[np.float64]:
        return self.compute()


__all__ = [
    "HillshadeCalculator",
    "triangle_hillshade",
    "DEFAULT_AZIMUTH",
    "DEFAULT_ALTITUDE",
]
