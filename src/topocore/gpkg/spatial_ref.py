"""
topocore.gpkg.spatial_ref
=============================

Resolves an EPSG code into everything `gpkg_spatial_ref_sys` needs,
using `pyproj` (already a mandatory TopoCore dependency) rather than
guessing at WKT by hand.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS
from pyproj.exceptions import CRSError

from topocore.gpkg.exceptions import GPKGExportError

#: GeoPackage-spec placeholder SRS ids -- explicitly rejected, since
#: TopoCore requires a real, resolvable CRS (see GPKGExportOptions.epsg).
_UNDEFINED_SRS_IDS = frozenset({-1, 0})


@dataclass(frozen=True, slots=True)
class ResolvedSRS:
    srs_id: int
    srs_name: str
    organization: str
    organization_coordsys_id: int
    definition: str
    description: str | None = None


def resolve_srs(epsg: int) -> ResolvedSRS:
    """
    Raises
    ------
    GPKGExportError
        If `epsg` is one of the GeoPackage 'undefined' placeholder
        ids (-1, 0), or if `pyproj` doesn't recognize it as a real
        EPSG code.
    """
    if epsg in _UNDEFINED_SRS_IDS:
        raise GPKGExportError(
            f"epsg={epsg} is a GeoPackage 'undefined' placeholder SRS, not a real "
            "coordinate reference system. GeoPackage export requires an explicit, "
            "resolvable EPSG code."
        )

    try:
        crs = CRS.from_epsg(epsg)
    except CRSError as exc:
        raise GPKGExportError(f"EPSG:{epsg} is not a recognized coordinate reference system.") from exc

    return ResolvedSRS(
        srs_id=epsg,
        srs_name=crs.name,
        organization="EPSG",
        organization_coordsys_id=epsg,
        definition=crs.to_wkt(),
        description=None,
    )


__all__ = ["ResolvedSRS", "resolve_srs"]
