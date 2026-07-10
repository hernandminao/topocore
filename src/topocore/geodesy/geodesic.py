"""
topocore.geodesy.geodesic
=========================

Geodesic calculations over an ellipsoid.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geodesy import _cache

from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import GeodesicError


class GeodesicCalculator:
    """
    Calculates geodesic distances, azimuths, and areas
    using the CRS ellipsoid.
    """

    __slots__ = ("_geod",)

    def __init__(self, crs: CRS) -> None:
        try:
            ellipsoid = crs.ellipsoid
	    
            if ellipsoid is None:
                raise GeodesicError(
                    "CRS does not have an ellipsoid."
                )
            
            self._geod = _cache.get_geod(
                ellipsoid.semi_major_axis,
                ellipsoid.inverse_flattening,
            )

        except Exception as exc:
            raise GeodesicError(
                "Failed to initialize Geodesic calculator."
            ) from exc

    def distance(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> float:
        """
        Calculate the geodesic distance in meters.
        """
        try:
            _, _, distance = self._geod.inv(
                lon1,
                lat1,
                lon2,
                lat2,
            )
            return float(distance)

        except Exception as exc:
            raise GeodesicError(
                "Geodesic distance calculation failed."
            ) from exc

    def azimuth(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> float:
        """
        Calculate the forward azimuth in degrees.
        """
        try:
            azimuth, _, _ = self._geod.inv(
                lon1,
                lat1,
                lon2,
                lat2,
            )

            return float(azimuth)

        except Exception as exc:
            raise GeodesicError(
                "Geodesic azimuth calculation failed."
            ) from exc

    def inverse(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> tuple[float, float, float]:
        """
        Compute inverse geodesic solution.

        Returns
        -------
        tuple
            Forward azimuth,
            back azimuth,
            distance (meters).
        """
        try:
            az1, az2, distance = self._geod.inv(
                lon1,
                lat1,
                lon2,
                lat2,
            )

            return (
                float(az1),
                float(az2),
                float(distance),
            )

        except Exception as exc:
            raise GeodesicError(
                "Geodesic inverse calculation failed."
            ) from exc

    def forward(
        self,
        lon: float,
        lat: float,
        azimuth: float,
        distance: float,
    ) -> tuple[float, float, float]:
        """
        Compute forward geodesic solution.

        Returns
        -------
        tuple
            Longitude,
            latitude,
            back azimuth.
        """
        try:
            lon2, lat2, back_azimuth = self._geod.fwd(
                lon,
                lat,
                azimuth,
                distance,
            )

            return (
                float(lon2),
                float(lat2),
                float(back_azimuth),
            )

        except Exception as exc:
            raise GeodesicError(
                "Geodesic forward calculation failed."
            ) from exc

    def polygon_area(
        self,
        lons: list[float],
        lats: list[float],
    ) -> float:
        """
        Calculate polygon area on the ellipsoid.

        Returns
        -------
        float
            Area in square meters.
        """
        try:
            area, _ = self._geod.polygon_area_perimeter(
                lons,
                lats,
            )

            return float(abs(area))

        except Exception as exc:
            raise GeodesicError(
                "Geodesic polygon area calculation failed."
            ) from exc


__all__ = [
    "GeodesicCalculator",
]