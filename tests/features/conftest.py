"""
Shared fixtures for topocore.geodesy.vertical tests.

Builds real, valid GeoTIFF files via GDAL itself -- never a
hand-rolled binary format guess -- with hand-computable known values,
so bilinear interpolation can be verified against arithmetic, not
just "it runs". No real geoid model data (EGM96/EGM2008 or otherwise)
is bundled or fabricated anywhere in this test suite; every grid here
is a small, clearly synthetic fixture used purely to verify the
loading/interpolation/safety machinery itself.

Confirmed directly, and noted here for anyone maintaining these
tests: this environment's own GDAL Python bindings were compiled
against NumPy 1.x while NumPy 2.x is installed. Raster IO in
topocore.geodesy.vertical.geoid_grid (and these fixtures) uses
`Band.ReadRaster()`/`WriteRaster()` with `struct` specifically to
avoid GDAL's own `gdal_array` (NumPy) bridge, which is the part that
doesn't work in this environment -- this is not merely a style
choice, it's a compatibility requirement here.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import pytest


def _write_geotiff(
    path: Path,
    values: list[list[float]],
    *,
    origin_x: float,
    origin_y: float,
    pixel_size: float,
    nodata: float | None = None,
) -> None:
    from osgeo import gdal, osr  # type: ignore[import-untyped]

    rows = len(values)
    columns = len(values[0])

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), columns, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((origin_x, pixel_size, 0.0, origin_y, 0.0, -pixel_size))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())

    band = dataset.GetRasterBand(1)
    flat = [v for row in values for v in row]
    band.WriteRaster(
        0,
        0,
        columns,
        rows,
        struct.pack(f"={len(flat)}f", *flat),
        buf_type=gdal.GDT_Float32,
    )

    if nodata is not None:
        band.SetNoDataValue(nodata)

    dataset.FlushCache()


@pytest.fixture
def synthetic_geoid_path(tmp_path: Path) -> Iterator[Path]:
    """
    A 3x3 grid, lon [0,2] x lat [0,2], resolution 1 degree.

    Grid points (row, col) -> (lon, lat) -> value:
        (0,0)->(0,2)=10   (0,1)->(1,2)=20   (0,2)->(2,2)=30
        (1,0)->(0,1)=40   (1,1)->(1,1)=50   (1,2)->(2,1)=60
        (2,0)->(0,0)=70   (2,1)->(1,0)=80   (2,2)->(2,0)=90

    Hand-computed interpolation check used throughout this suite:
    at (0.5, 1.5), bilinear interpolation is the average of the 4
    surrounding grid points (0,2)=10, (1,2)=20, (0,1)=40, (1,1)=50
    -> (10+20+40+50)/4 = 30.0 exactly.
    """
    path = tmp_path / "synthetic_geoid.tif"
    values = [
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0],
    ]
    _write_geotiff(path, values, origin_x=0.0, origin_y=2.0, pixel_size=1.0)
    yield path


@pytest.fixture
def geoid_with_nodata_path(tmp_path: Path) -> Iterator[Path]:
    """A 2x2 grid with one nodata cell, for verifying the nodata safety check."""
    path = tmp_path / "geoid_with_nodata.tif"
    values = [
        [10.0, -9999.0],
        [30.0, 40.0],
    ]
    _write_geotiff(path, values, origin_x=0.0, origin_y=1.0, pixel_size=1.0, nodata=-9999.0)
    yield path


@pytest.fixture
def multiband_geotiff_path(tmp_path: Path) -> Iterator[Path]:
    """A valid GeoTIFF, but with 2 bands -- must be rejected (a geoid grid needs exactly 1)."""
    from osgeo import gdal, osr  # type: ignore[import-untyped]

    path = tmp_path / "multiband.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), 2, 2, 2, gdal.GDT_Float32)
    dataset.SetGeoTransform((0.0, 1.0, 0.0, 1.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.FlushCache()
    yield path


@pytest.fixture
def colombia_extent_geoid_path(tmp_path: Path) -> Iterator[Path]:
    """
    A deliberately NON-square, asymmetric 4x3 grid over Colombia's
    own real geographic extent (lon -80 to -66, lat -4 to 13) with a
    UNIQUE value at every one of the 12 grid points -- constructed
    specifically to catch any row/column transposition or north/south
    or east/west axis inversion bug, which a symmetric grid (like
    `synthetic_geoid_path`'s own 3x3) could not reveal.

    Values increase toward the east (higher column index) and toward
    the south (higher row index), by construction:
        row 0 (lat=13,  north): [1, 2, 3, 4]
        row 1 (lat mid)      : [10, 20, 30, 40]
        row 2 (lat=-4,  south): [100, 200, 300, 400]

    Real-data caveat, stated plainly: no genuine EGM96/EGM2008 grid
    file (or live geoid-height lookup service) is reachable from this
    development environment -- confirmed directly, every path leads
    to a domain outside this environment's network allowlist
    (cdn.proj.org for PROJ's own grids; earth-info.NGA.mil and
    NGS.NOAA.gov, both referenced by the real `pygeodesy` package's
    own documentation for obtaining real data or a live lookup).
    Colombia's own real geographic extent is used here (a public,
    unlicensed geographic fact, not the licensed undulation data
    itself) specifically to make this axis/coverage verification
    meaningful rather than arbitrary.
    """
    path = tmp_path / "colombia_test_grid.tif"
    from osgeo import gdal, osr  # type: ignore[import-untyped]

    values = [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [100.0, 200.0, 300.0, 400.0],
    ]
    pixel_width = (-66.0 - (-80.0)) / 4
    pixel_height = (13.0 - (-4.0)) / 3

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), 4, 3, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((-80.0, pixel_width, 0.0, 13.0, 0.0, -pixel_height))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    flat = [v for row in values for v in row]
    band.WriteRaster(0, 0, 4, 3, struct.pack(f"={len(flat)}f", *flat), buf_type=gdal.GDT_Float32)
    band.SetNoDataValue(-9999.0)
    dataset.FlushCache()
    yield path


@pytest.fixture
def sub_degree_resolution_geoid_path(tmp_path: Path) -> Iterator[Path]:
    """
    A 3x3 grid at 2.5 arc-minute resolution (2.5/60 degrees per
    cell) -- matching real EGM2008's own coarse-grid convention --
    over a tiny real extent, with realistic undulation magnitudes
    (-30 to +50 m, within the real-world ~-100 to +100 m range for
    EGM96/2008, confirmed via public geoid documentation), rather
    than the simplified round numbers used elsewhere in this suite.
    """
    path = tmp_path / "sub_degree_geoid.tif"
    resolution = 2.5 / 60.0
    values = [
        [-30.5, -20.3, -10.1],
        [0.0, 10.2, 20.4],
        [30.6, 40.8, 50.0],
    ]
    _write_geotiff(path, values, origin_x=0.0, origin_y=resolution * 3, pixel_size=resolution)
    yield path


@pytest.fixture
def negative_pixel_width_geoid_path(tmp_path: Path) -> Iterator[Path]:
    """
    A 3x3 grid with origin at the northeast corner and a NEGATIVE
    pixel_width (columns increase westward) -- less common than the
    standard northwest-origin convention used elsewhere in this
    suite, but a structurally valid GeoTransform.
    """
    from osgeo import gdal, osr  # type: ignore[import-untyped]

    path = tmp_path / "negative_pixel_width.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), 3, 3, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((3.0, -1.0, 0.0, 3.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    values = [float(i) for i in range(9)]
    band.WriteRaster(0, 0, 3, 3, struct.pack(f"={len(values)}f", *values), buf_type=gdal.GDT_Float32)
    dataset.FlushCache()
    yield path
