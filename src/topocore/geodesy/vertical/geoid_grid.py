"""
topocore.geodesy.vertical.geoid_grid
=======================================

Loads a real geodetic grid (GeoTIFF -- the format PROJ itself
recommends for geodetic grids today; see PROJ's own "Geodetic TIFF
grids" specification) and interpolates a geoid undulation value `N`
at an arbitrary `(longitude, latitude)`.

This module does not include, bundle, or fabricate any real geoid
model data (EGM96, EGM2008, or otherwise). It reads whatever GeoTIFF
file you give it -- confirmed directly, this codebase's own
development environment has no such file and no network path to
obtain one (see the project's own geodesy documentation). Bring your
own real geoid grid to use this for real work; what's implemented and
tested here is the loading and interpolation machinery itself,
verified against a synthetic grid with hand-computed expected values.

Why GDAL, and why lazily imported
----------------------------------
GDAL is PROJ's own recommended tool for reading geodetic grid files
(the legacy `.gtx` binary format has been superseded by a GeoTIFF
profile that requires genuine TIFF/GeoTIFF parsing to read
correctly -- not something to hand-roll from a byte-level guess at
the spec, since a subtly wrong binary parser would silently produce
wrong undulation values, exactly the failure mode this whole package
exists to prevent). GDAL is a heavier dependency than the rest of
`topocore.geodesy` (which only needs `pyproj`), so it is imported
lazily inside `GeoidGrid.from_geotiff()` -- matching the same
optional-dependency pattern already established elsewhere in this
codebase (PyYAML for catalog loading, xgboost/lightgbm for ML
classifiers): importing `topocore.geodesy.vertical` itself never
requires GDAL to be installed, only actually loading a grid does.

A confirmed, environment-specific caveat: this development
environment's own GDAL Python bindings were compiled against NumPy
1.x, while NumPy 2.x is installed -- calling `gdal.DontUseExceptions()`
or any GDAL codepath that reaches for its own `gdal_array` (NumPy)
bridge prints a harmless-but-noisy traceback to stderr. Confirmed
directly this does not affect correctness. This module therefore
reads/writes raster data via `Band.ReadRaster()`/`WriteRaster()`
(raw bytes, unpacked with the standard library's own `struct` module)
rather than `ReadAsArray()`/`WriteArray()`, sidestepping the
`gdal_array` bridge entirely -- this is not merely a workaround for
that one warning, it also means this module has no NumPy-ABI
compatibility requirement of its own with whatever GDAL build a
deployment environment happens to have.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from topocore.geodesy.exceptions import CRSError
from topocore.geodesy.vertical.exceptions import GeoidError, MissingGeoidGridError

_GEO_TRANSFORM_TOLERANCE: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class GeoidGrid:
    """
    An immutable, in-memory geodetic grid: a regular longitude/
    latitude grid of scalar values (geoid undulation, typically in
    meters) plus the georeferencing needed to interpolate at an
    arbitrary point.

    Constructed via `GeoidGrid.from_geotiff()` -- never directly, the
    same "no public constructor, use a named factory" convention
    already established by `topocore.geodesy.CRS`.
    """

    origin_longitude: float
    origin_latitude: float
    pixel_width: float
    pixel_height: float
    columns: int
    rows: int
    _values: tuple[float, ...]
    nodata: float | None
    source_path: str

    @classmethod
    def from_geotiff(cls, path: str | Path) -> GeoidGrid:
        """
        Load a single-band GeoTIFF as a `GeoidGrid`.

        Raises
        ------
        MissingGeoidGridError
            If `path` does not exist.
        GeoidError
            If GDAL is not installed, or the file exists but cannot
            be opened as a valid single-band raster.
        """
        resolved_path = Path(path)

        if not resolved_path.is_file():
            raise MissingGeoidGridError(f"Geoid grid file not found: '{resolved_path}'.")

        try:
            from osgeo import gdal
        except ImportError as exc:
            raise GeoidError(
                "Reading a geoid grid requires GDAL, which is not installed. "
                "Install it with `pip install gdal` (matching your system's "
                "libgdal version) to use GeoidGrid.from_geotiff()."
            ) from exc

        from topocore.geodesy.crs import CRS

        # Deliberately NOT calling gdal.UseExceptions(): confirmed
        # directly that doing so triggers GDAL's own attempt to also
        # load its gdal_array (NumPy) bridge, which prints a
        # harmless-but-noisy traceback to stderr in this environment
        # (NumPy 1.x-compiled bindings vs. NumPy 2.x installed -- see
        # this module's own docstring). Every failure path below is
        # already checked explicitly (`dataset is None`,
        # `RasterCount != 1`, ...) without relying on GDAL's own
        # exception mechanism at all, so UseExceptions() buys nothing
        # here.
        dataset = gdal.Open(str(resolved_path))
        if dataset is None:
            raise GeoidError(f"Failed to open '{resolved_path}' as a GDAL raster.")

        if dataset.RasterCount != 1:
            raise GeoidError(
                f"'{resolved_path}' has {dataset.RasterCount} raster bands; "
                "a geoid grid must have exactly 1 (the undulation values)."
            )

        # Found and fixed during this project's own documentation
        # audit: this used to accept any single-band raster's own
        # GeoTransform origin/pixel values unconditionally, treating
        # them as longitude/latitude in degrees with no check at all
        # -- a projected grid (e.g. in UTM meters) would have been
        # silently misinterpreted as geographic, producing a wrong
        # undulation with no error, the exact failure mode this
        # module's own docstring states it exists to prevent for the
        # binary-parsing side. Confirmed reachable: nothing anywhere
        # in this codebase previously validated the file's own CRS
        # before this fix. Now confirmed explicitly geographic before
        # accepting the file, reusing `topocore.geodesy.CRS` (already
        # a mandatory dependency via pyproj, unlike GDAL) rather than
        # inventing a second WKT-parsing path.
        projection_wkt = dataset.GetProjection()
        if not projection_wkt:
            raise GeoidError(
                f"'{resolved_path}' has no coordinate reference system information. "
                "A geoid grid must be in geographic (longitude/latitude) coordinates, "
                "and this cannot be confirmed without a CRS in the file itself."
            )

        try:
            grid_crs = CRS.from_wkt(projection_wkt)
        except CRSError as exc:
            raise GeoidError(f"'{resolved_path}' has an unreadable coordinate reference system.") from exc

        if not grid_crs.is_geographic:
            raise GeoidError(
                f"'{resolved_path}' is not in geographic (longitude/latitude) coordinates "
                f"(its own CRS is '{grid_crs.name}'). A geoid grid's own GeoTransform origin "
                "and pixel size must be in degrees for undulation_at(longitude, latitude) to "
                "give a correct result -- reproject the file to a geographic CRS first."
            )

        origin_x, pixel_width, x_skew, origin_y, y_skew, pixel_height = dataset.GetGeoTransform()

        if not math.isclose(
            x_skew,
            0.0,
            abs_tol=_GEO_TRANSFORM_TOLERANCE,
        ) or not math.isclose(
            y_skew,
            0.0,
            abs_tol=_GEO_TRANSFORM_TOLERANCE,
        ):
            raise GeoidError(f"'{resolved_path}' has a rotated/sheared GeoTransform, which is not supported.")

        band = dataset.GetRasterBand(1)
        columns, rows = band.XSize, band.YSize
        nodata = band.GetNoDataValue()

        raw = band.ReadRaster(0, 0, columns, rows, buf_type=gdal.GDT_Float32)
        # Native byte order ("=", not "<"), matching GDAL's own
        # documented convention and confirmed directly: GDAL's own
        # official raster API tutorial reads scanlines with plain
        # struct.unpack('f' * xsize, ...) -- no explicit byte-order
        # prefix, i.e. native order -- and a separate GDAL binding's
        # own documentation states plainly "read_band returns raw
        # bytes in native endianness". A hardcoded "<" (little-endian)
        # would silently misread every value on a genuine big-endian
        # machine; this reads correctly regardless of platform.
        values = struct.unpack(f"={columns * rows}f", raw)

        return cls(
            origin_longitude=origin_x,
            origin_latitude=origin_y,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            columns=columns,
            rows=rows,
            _values=values,
            nodata=nodata,
            source_path=str(resolved_path),
        )

    def _value_at(self, row: int, column: int) -> float:
        return self._values[row * self.columns + column]

    def undulation_at(self, longitude: float, latitude: float) -> float:
        """
        Bilinearly interpolate the undulation value at
        `(longitude, latitude)`.

        Confirmed directly: this correctly handles a `pixel_width`
        that is itself negative (an origin at the grid's own
        northeast corner, with columns increasing westward) with no
        special-casing needed -- the same formula naturally
        generalizes, verified against a hand-computed expected value.

        **Caller responsibility, reassuring side effect**: this
        method never validates that `longitude`/`latitude` are
        genuinely geographic degrees rather than projected
        coordinates (see `topocore.geodesy.vertical.transform`'s own
        stated caller responsibility) -- but confirmed directly,
        passing projected coordinates (typically many orders of
        magnitude larger than any real geoid grid's own geographic
        extent) naturally fails the extent check below rather than
        silently producing a nonsensical interpolated value. This is
        an incidental consequence of the extent check, not a
        deliberate CRS-type detection mechanism.

        Raises
        ------
        MissingGeoidGridError
            If the point falls outside this grid's own extent, or
            any of the 4 surrounding grid points is the grid's own
            nodata value -- both are "no usable geoid data here"
            cases, never silently approximated.
        """
        column_f = (longitude - self.origin_longitude) / self.pixel_width
        row_f = (latitude - self.origin_latitude) / self.pixel_height

        column0 = int(column_f)
        row0 = int(row_f)

        if column_f < 0 or row_f < 0 or column0 >= self.columns - 1 or row0 >= self.rows - 1:
            raise MissingGeoidGridError(
                f"Point ({longitude}, {latitude}) is outside the geoid grid's own "
                f"extent (source: '{self.source_path}')."
            )

        column_t = column_f - column0
        row_t = row_f - row0

        corners = (
            self._value_at(row0, column0),
            self._value_at(row0, column0 + 1),
            self._value_at(row0 + 1, column0),
            self._value_at(row0 + 1, column0 + 1),
        )

        if self.nodata is not None and any(corner == self.nodata for corner in corners):
            raise MissingGeoidGridError(
                f"Point ({longitude}, {latitude}) falls in or next to a nodata region "
                f"of the geoid grid (source: '{self.source_path}')."
            )

        top = corners[0] + column_t * (corners[1] - corners[0])
        bottom = corners[2] + column_t * (corners[3] - corners[2])

        return top + row_t * (bottom - top)


__all__ = ["GeoidGrid"]
