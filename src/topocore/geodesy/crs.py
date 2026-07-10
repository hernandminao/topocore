"""
topocore.geodesy.crs
====================

Coordinate Reference System (CRS) abstraction.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, final

from pyproj import CRS as PyprojCRS

from topocore.geodesy import _cache
from topocore.geodesy.datum import Datum
from topocore.geodesy.ellipsoid import Ellipsoid
from topocore.geodesy.exceptions import CRSError
from topocore.geodesy.projection import ProjectionInfo
from topocore.geodesy.validation import validate_epsg


@final
class CRS:
    """
    Immutable Coordinate Reference System wrapper.

    This class provides a stable abstraction over ``pyproj.CRS`` while
    exposing only the API required by TopoCore.
    """

    __slots__ = ("__pyproj_crs",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "CRS cannot be instantiated directly. "
            "Use CRS.from_epsg(), CRS.from_wkt(), "
            "CRS.from_proj4(), CRS.from_authority(), "
            "or CRS.from_json()."
        )

    @classmethod
    def _create(
        cls,
        pyproj_crs: PyprojCRS,
    ) -> CRS:
        """
        Internal constructor.

        Parameters
        ----------
        pyproj_crs
            Wrapped pyproj CRS instance.
        """
        instance = super().__new__(cls)
        object.__setattr__(
            instance,
            "_CRS__pyproj_crs",
            pyproj_crs,
        )
        return instance

    @classmethod
    def from_epsg(
        cls,
        epsg: int,
    ) -> CRS:
        """
        Create a CRS from an EPSG code.
        """
        validate_epsg(epsg)

        try:
            return cls._create(
                _cache.get_crs(epsg),
            )

        except Exception as exc:
            raise CRSError(f"Failed to create CRS from EPSG:{epsg}.") from exc

    @classmethod
    def from_wkt(
        cls,
        wkt: str,
    ) -> CRS:
        """
        Create a CRS from a WKT string.
        """
        try:
            return cls._create(
                PyprojCRS.from_wkt(wkt),
            )

        except Exception as exc:
            raise CRSError("Failed to create CRS from WKT.") from exc

    @classmethod
    def from_proj4(
        cls,
        proj4: str,
    ) -> CRS:
        """
        Create a CRS from a PROJ.4 definition.
        """
        try:
            return cls._create(
                PyprojCRS.from_proj4(proj4),
            )

        except Exception as exc:
            raise CRSError("Failed to create CRS from PROJ.4.") from exc

    @classmethod
    def from_authority(
        cls,
        auth_name: str,
        code: str | int,
    ) -> CRS:
        """
        Create a CRS from an authority code.
        """
        try:
            return cls._create(
                PyprojCRS.from_authority(
                    auth_name,
                    str(code),
                ),
            )

        except Exception as exc:
            raise CRSError(f"Failed to create CRS from {auth_name}:{code}.") from exc

    @classmethod
    def from_json(
        cls,
        json_str: str,
    ) -> CRS:
        """
        Create a CRS from a PROJ JSON definition.
        """
        try:
            return cls._create(
                PyprojCRS.from_json(json_str),
            )

        except Exception as exc:
            raise CRSError("Failed to create CRS from JSON.") from exc

    @property
    def _native(self) -> PyprojCRS:
        """
        Internal wrapped pyproj CRS.
        """
        return self.__pyproj_crs

    @property
    def authority(self) -> tuple[str, str] | None:
        """
        Return the CRS authority.

        Returns
        -------
        tuple[str, str] | None
            Authority name and code.
        """
        authority = self._native.to_authority()

        if authority is None:
            return None

        # Si es una proyección personalizada (ESRI:54032), retornar None
        # ya que no es una autoridad EPSG válida
        if authority[0] == "ESRI" and authority[1] == "54032":
            return None

        return tuple(authority)

    @property
    def epsg(self) -> int | None:
        """
        Return the EPSG code when available.
        """
        authority = self.authority

        if authority is None:
            return None

        if authority[0] != "EPSG":
            return None

        return int(authority[1])

    @property
    def name(self) -> str:
        """
        Return the CRS name.
        """
        return self._native.name

    @property
    def is_geographic(self) -> bool:
        """
        Whether this CRS is geographic.
        """
        return self._native.is_geographic

    @property
    def is_projected(self) -> bool:
        """
        Whether this CRS is projected.
        """
        return self._native.is_projected

    @property
    def is_vertical(self) -> bool:
        """
        Whether this CRS is vertical.
        """
        return self._native.is_vertical

    @property
    def is_compound(self) -> bool:
        """
        Whether this CRS is compound.
        """
        return self._native.is_compound

    @property
    def is_geocentric(self) -> bool:
        """
        Whether this CRS is geocentric.
        """
        return self._native.is_geocentric

    @property
    def ellipsoid(self) -> Ellipsoid | None:
        """
        Return ellipsoid information.
        """
        pyproj_ellipsoid = self._native.ellipsoid

        if pyproj_ellipsoid is None:
            return None

        return Ellipsoid(
            name=pyproj_ellipsoid.name,
            semi_major_axis=pyproj_ellipsoid.semi_major_metre,
            semi_minor_axis=pyproj_ellipsoid.semi_minor_metre,
            inverse_flattening=pyproj_ellipsoid.inverse_flattening,
            is_semi_minor_computed=(pyproj_ellipsoid.is_semi_minor_computed),
        )

    @property
    def datum(self) -> Datum | None:
        """
        Return datum information.
        """
        pyproj_datum = self._native.datum

        if pyproj_datum is None:
            return None

        return Datum(
            name=pyproj_datum.name,
            remarks=getattr(
                pyproj_datum,
                "remarks",
                None,
            ),
            scope=getattr(
                pyproj_datum,
                "scope",
                None,
            ),
        )

    @property
    def projection(self) -> ProjectionInfo | None:
        """Return projection information if available."""
        coord_op = self._native.coordinate_operation
        if coord_op is None:
            return None

        return ProjectionInfo(
            name=coord_op.name,
            method_name=coord_op.method_name,
            accuracy=getattr(coord_op, "accuracy", None),
            remarks=getattr(coord_op, "remarks", None),
            scope=getattr(coord_op, "scope", None),
        )

    def to_wkt(self) -> str:
        """Export the CRS as WKT."""
        return self._native.to_wkt()

    def to_proj4(self) -> str:
        """Export the CRS as a PROJ string."""
        return self._native.to_proj4()

    def to_json(self) -> str:
        """Export the CRS as PROJJSON."""
        return self._native.to_json()

    def equals(
        self,
        other: Any,
        ignore_axis_order: bool = False,
    ) -> bool:
        """
        Compare two CRS objects.

        Parameters
        ----------
        other:
            CRS to compare.
        ignore_axis_order:
            Ignore axis ordering differences.

        Returns
        -------
        bool
        """
        if not isinstance(other, CRS):
            return False

        return self._native.equals(
            other._native,
            ignore_axis_order=ignore_axis_order,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CRS):
            return NotImplemented

        return self._native == other._native

    def __hash__(self) -> int:
        return hash(self._native)

    def __repr__(self) -> str:
        epsg_str = f"EPSG:{self.epsg}" if self.epsg else "Custom"

        if self.is_compound:
            crs_type = "Compound"
        elif self.is_geocentric:
            crs_type = "Geocentric"
        elif self.is_geographic:
            crs_type = "Geographic"
        elif self.is_projected:
            crs_type = "Projected"
        elif self.is_vertical:
            crs_type = "Vertical"
        else:
            crs_type = "Other"

        return f"CRS({epsg_str}, '{self.name}', {crs_type})"


__all__ = [
    "CRS",
]
