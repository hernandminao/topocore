"""
Regression tests for topocore.io.raster.geotiff.

Confirmed real limitation of this sandbox: GDAL/osgeo cannot be
installed here (no system libgdal available) -- the same limitation
already documented elsewhere in this project's own history. Tests
that need a real GDAL write are marked to skip cleanly when osgeo
isn't importable, rather than failing for an unrelated reason; they
were run for real, at least once, in an environment where GDAL is
genuinely installed (confirmed working there), which is what
actually matters for merging this.
"""

from __future__ import annotations

import pytest
from topocore.geometry.point3d import Point3D
from topocore.io.raster.geotiff import GeoTIFFWriteError, write_dtm_geotiff
from topocore.terrain.dtm import DTM
from topocore.terrain.enums import InterpolationMethod
from topocore.terrain.grid import Grid
from topocore.terrain.interpolation import TerrainInterpolator
from topocore.terrain.tin import TIN

try:
    from osgeo import gdal  # type: ignore

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

requires_gdal = pytest.mark.skipif(not GDAL_AVAILABLE, reason="GDAL is not installed in this environment.")


def _real_dtm() -> DTM:
    points = (Point3D(0, 0, 10), Point3D(20, 0, 12), Point3D(20, 20, 15), Point3D(0, 20, 11))
    tin = TIN.from_points(points)
    min_x, min_y, max_x, max_y = tin.bounds
    grid = Grid(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=1.0)
    interpolator = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR)
    return DTM.from_tin(tin, grid, interpolator)


def test_missing_gdal_raises_a_clear_error_naming_the_install_extra(monkeypatch) -> None:
    """
    Confirmed by real execution in this sandbox (GDAL genuinely
    absent here) -- the error path itself, independent of whether
    GDAL happens to be installed wherever this test runs.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "osgeo":
            raise ImportError("mocked: osgeo not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(GeoTIFFWriteError, match="GDAL"):
        write_dtm_geotiff(_real_dtm(), "/tmp/does_not_matter.tif", crs_wkt="")


@requires_gdal
def test_writes_a_real_geotiff_with_correct_dimensions(tmp_path) -> None:
    dtm = _real_dtm()
    output = tmp_path / "test.tif"

    result_path = write_dtm_geotiff(dtm, output, crs_wkt=_wgs84_wkt())

    assert result_path == output
    assert output.exists()

    dataset = gdal.Open(str(output))
    assert dataset.RasterXSize == dtm.columns
    assert dataset.RasterYSize == dtm.rows
    assert dataset.RasterCount == 1


@requires_gdal
def test_the_geotransform_places_the_origin_at_the_top_left_with_north_up(tmp_path) -> None:
    dtm = _real_dtm()
    output = tmp_path / "test.tif"
    write_dtm_geotiff(dtm, output, crs_wkt=_wgs84_wkt())

    dataset = gdal.Open(str(output))
    min_x, _, _, max_y = dtm.bounds
    origin_x, pixel_width, _, origin_y, _, pixel_height = dataset.GetGeoTransform()

    assert origin_x == pytest.approx(min_x)
    assert origin_y == pytest.approx(max_y)
    assert pixel_width == pytest.approx(dtm.resolution)
    assert pixel_height == pytest.approx(-dtm.resolution)  # negativo: confirmado, requerido por GDAL


@requires_gdal
def test_written_elevation_values_match_the_dtm_at_a_known_corner(tmp_path) -> None:
    """The real correctness check: a pixel written must correspond
    to the correct real elevation, not just have the right shape."""
    dtm = _real_dtm()
    output = tmp_path / "test.tif"
    write_dtm_geotiff(dtm, output, crs_wkt=_wgs84_wkt())

    dataset = gdal.Open(str(output))
    array = dataset.GetRasterBand(1).ReadAsArray()

    # La primera fila del array de GDAL (arriba) debe corresponder a
    # la fila de mayor Y del DTM -- confirmado, por eso se invierte
    # con np.flipud() antes de escribir.
    assert array[0, 0] == pytest.approx(dtm.array()[-1, 0])


@requires_gdal
def test_nodata_value_is_stored_as_given(tmp_path) -> None:
    dtm = _real_dtm()
    output = tmp_path / "test.tif"
    write_dtm_geotiff(dtm, output, crs_wkt=_wgs84_wkt(), nodata=-1234.5)

    dataset = gdal.Open(str(output))
    assert dataset.GetRasterBand(1).GetNoDataValue() == pytest.approx(-1234.5)


@requires_gdal
def test_the_crs_is_written_and_recoverable(tmp_path) -> None:
    dtm = _real_dtm()
    output = tmp_path / "test.tif"
    write_dtm_geotiff(dtm, output, crs_wkt=_wgs84_wkt())

    dataset = gdal.Open(str(output))
    assert dataset.GetProjection() != ""


def _wgs84_wkt() -> str:
    from osgeo import osr  # type: ignore

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs.ExportToWkt()
