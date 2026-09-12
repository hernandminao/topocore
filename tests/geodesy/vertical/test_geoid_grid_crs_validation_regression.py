"""
Regression test for a real defect found and fixed while writing this
project's own usage documentation (24-usage/geodesy.md):
`GeoidGrid.from_geotiff()` used to accept any single-band raster's
own `GeoTransform` origin/pixel values unconditionally, treating them
as longitude/latitude in degrees with no check at all. A projected
geoid grid (e.g. in UTM meters) would have been silently
misinterpreted as geographic, producing a wrong undulation value with
no error -- the exact "silently wrong, not loudly failing" shape this
module's own docstring already states it exists to prevent for the
binary-parsing side, and the same philosophy `VerticalTransformer`
itself already follows (raising `MissingGeoidGridError` rather than
silently returning an uncorrected height).

Fixed by reading the raster's own embedded CRS (`dataset.GetProjection()`)
and confirming it is genuinely geographic via `topocore.geodesy.CRS`
(already a mandatory dependency via pyproj) before accepting the file.

GDAL's own native library (`libgdal`) could not be installed in this
audit's own sandbox environment (no `gdal-config`, and a system-level
`apt-get install libgdal-dev` did not complete). These tests therefore
mock `osgeo.gdal` directly via `sys.modules`, exercising the real,
unmodified `from_geotiff()` code path -- including this fix -- rather
than skipping verification entirely. The CRS-validation logic itself
was additionally confirmed directly against real `CRS` objects
(EPSG:4326 vs. EPSG:32618) during this same fix's own development.
"""

from __future__ import annotations

import struct
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from topocore.geodesy.crs import CRS
from topocore.geodesy.vertical.exceptions import GeoidError


def _install_fake_gdal(*, projection_wkt: str, geo_transform: tuple[float, ...] = (-75.0, 0.01, 0.0, 5.0, 0.0, -0.01)):
    """
    Installs a fake `osgeo.gdal` module in `sys.modules`, so the real
    `from osgeo import gdal` line inside `from_geotiff()` resolves to
    a controllable fake instead of requiring a real GDAL install.
    """
    values = (1.0, 2.0, 3.0, 4.0)
    raw = struct.pack(f"={len(values)}f", *values)

    fake_band = MagicMock()
    fake_band.XSize = 2
    fake_band.YSize = 2
    fake_band.GetNoDataValue.return_value = None
    fake_band.ReadRaster.return_value = raw

    fake_dataset = MagicMock()
    fake_dataset.RasterCount = 1
    fake_dataset.GetProjection.return_value = projection_wkt
    fake_dataset.GetGeoTransform.return_value = geo_transform
    fake_dataset.GetRasterBand.return_value = fake_band

    fake_gdal = types.ModuleType("gdal")
    fake_gdal.Open = MagicMock(return_value=fake_dataset)  # type: ignore[attr-defined]
    fake_gdal.GDT_Float32 = 6  # type: ignore[attr-defined]

    fake_osgeo = types.ModuleType("osgeo")
    fake_osgeo.gdal = fake_gdal  # type: ignore[attr-defined]  # dynamic test double

    sys.modules["osgeo"] = fake_osgeo
    sys.modules["osgeo.gdal"] = fake_gdal


@pytest.fixture(autouse=True)
def _cleanup_fake_gdal():
    yield
    sys.modules.pop("osgeo", None)
    sys.modules.pop("osgeo.gdal", None)


def test_from_geotiff_accepts_a_genuinely_geographic_crs(tmp_path: Path) -> None:
    from topocore.geodesy.vertical.geoid_grid import GeoidGrid

    wgs84_wkt = CRS.from_epsg(4326).to_wkt()
    _install_fake_gdal(projection_wkt=wgs84_wkt)

    path = tmp_path / "geoid.tif"
    path.write_bytes(b"not a real tiff -- gdal.Open is mocked")

    grid = GeoidGrid.from_geotiff(path)

    assert grid.columns == 2
    assert grid.rows == 2


def test_from_geotiff_rejects_a_projected_crs(tmp_path: Path) -> None:
    """
    The exact real defect this fix addresses: before the fix, this
    exact scenario (a valid single-band raster, but in UTM meters,
    not degrees) was accepted silently, with no error at all.
    """
    from topocore.geodesy.vertical.geoid_grid import GeoidGrid

    utm18n_wkt = CRS.from_epsg(32618).to_wkt()
    _install_fake_gdal(projection_wkt=utm18n_wkt)

    path = tmp_path / "geoid_utm.tif"
    path.write_bytes(b"not a real tiff -- gdal.Open is mocked")

    with pytest.raises(GeoidError, match="not in geographic"):
        GeoidGrid.from_geotiff(path)


def test_from_geotiff_rejects_a_file_with_no_crs_at_all(tmp_path: Path) -> None:
    from topocore.geodesy.vertical.geoid_grid import GeoidGrid

    _install_fake_gdal(projection_wkt="")

    path = tmp_path / "geoid_no_crs.tif"
    path.write_bytes(b"not a real tiff -- gdal.Open is mocked")

    with pytest.raises(GeoidError, match="no coordinate reference system"):
        GeoidGrid.from_geotiff(path)


def test_from_geotiff_rejects_unreadable_wkt(tmp_path: Path) -> None:
    from topocore.geodesy.vertical.geoid_grid import GeoidGrid

    _install_fake_gdal(projection_wkt="this is not valid WKT at all")

    path = tmp_path / "geoid_bad_wkt.tif"
    path.write_bytes(b"not a real tiff -- gdal.Open is mocked")

    with pytest.raises(GeoidError, match="unreadable coordinate reference system"):
        GeoidGrid.from_geotiff(path)
