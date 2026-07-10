"""
topocore.geodesy.utm
====================

UTM zone resolution utilities.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import CRSError
from topocore.geodesy.validation import validate_lat_lon


_UTM_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"


@dataclass(frozen=True, slots=True)
class UTMZone:
    """
    Represents a UTM zone with detailed parameters.
    """

    zone_number: int
    zone_letter: str
    hemisphere: str
    epsg: int
    central_meridian: float
    false_easting: float
    false_northing: float

    @classmethod
    def from_latlon(
        cls,
        latitude: float,
        longitude: float,
    ) -> "UTMZone":
        """
        Determine the UTM zone for a latitude/longitude pair.
        """
        validate_lat_lon(latitude, longitude)

        zone_number = int((longitude + 180.0) / 6.0) + 1

        # Norway special case
        if 56.0 <= latitude < 64.0 and 3.0 <= longitude < 12.0:
            zone_number = 32

        # Svalbard special cases
        if 72.0 <= latitude <= 84.0 and longitude >= 0.0:
            if longitude < 9.0:
                zone_number = 31
            elif longitude < 21.0:
                zone_number = 33
            elif longitude < 33.0:
                zone_number = 35
            elif longitude < 42.0:
                zone_number = 37

        hemisphere = "N" if latitude >= 0.0 else "S"

        epsg = (
            32600 + zone_number
            if hemisphere == "N"
            else 32700 + zone_number
        )

        zone_letter = ""
        if -80.0 <= latitude <= 84.0:
            zone_letter = _UTM_LETTERS[int((latitude + 80.0) / 8.0)]

        return cls(
            zone_number=zone_number,
            zone_letter=zone_letter,
            hemisphere=hemisphere,
            epsg=epsg,
            central_meridian=(zone_number * 6) - 183,
            false_easting=500000.0,
            false_northing=(
                0.0
                if hemisphere == "N"
                else 10000000.0
            ),
        )

    @classmethod
    def from_epsg(
        cls,
        epsg: int,
    ) -> "UTMZone":
        """
        Resolve UTM information from an EPSG code.
        """
        if 32601 <= epsg <= 32660:
            zone = epsg - 32600
            return cls(
                zone_number=zone,
                zone_letter="",
                hemisphere="N",
                epsg=epsg,
                central_meridian=(zone * 6) - 183,
                false_easting=500000.0,
                false_northing=0.0,
            )

        if 32701 <= epsg <= 32760:
            zone = epsg - 32700
            return cls(
                zone_number=zone,
                zone_letter="",
                hemisphere="S",
                epsg=epsg,
                central_meridian=(zone * 6) - 183,
                false_easting=500000.0,
                false_northing=10000000.0,
            )

        raise CRSError(
            f"EPSG:{epsg} is not a standard UTM zone."
        )

    @classmethod
    def from_crs(
        cls,
        crs: CRS,
    ) -> "UTMZone | None":
        """
        Attempt to obtain the UTM zone from a CRS.
        """
        epsg = crs.epsg

        if epsg is None:
            return None

        try:
            return cls.from_epsg(epsg)
        except CRSError:
            return None


__all__ = [
    "UTMZone",
]