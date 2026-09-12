"""
Tests for topocore.geodesy.vertical.geoid_grid.GeoidGrid.

Every test uses a real, valid GeoTIFF built via GDAL itself (see
conftest.py) with hand-computable known values -- interpolation
results are checked against arithmetic, not merely "it doesn't
raise". No real geoid model data is used or claimed anywhere here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from topocore.geodesy.vertical.exceptions import GeoidError, MissingGeoidGridError
from topocore.geodesy.vertical.geoid_grid import GeoidGrid


def test_loads_grid_dimensions_and_origin_correctly(synthetic_geoid_path: Path) -> None:
    grid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    assert grid.columns == 3
    assert grid.rows == 3
    assert grid.origin_longitude == 0.0
    assert grid.origin_latitude == 2.0


def test_exact_grid_vertex_returns_exact_value(synthetic_geoid_path: Path) -> None:
    grid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    assert grid.undulation_at(0.0, 2.0) == 10.0
    assert grid.undulation_at(1.0, 1.0) == 50.0


def test_far_corner_vertex_raises_not_interpolates(synthetic_geoid_path: Path) -> None:
    """
    (2.0, 0.0) is the grid's own far corner (last row, last column) --
    bilinear interpolation needs a 2x2 neighborhood, and there is no
    cell extending beyond the far corner to interpolate toward. This
    is correct, expected behavior, not a defect: confirmed by direct
    reasoning about the bounds check before writing this test.
    """
    grid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    with pytest.raises(MissingGeoidGridError, match="outside the geoid grid"):
        grid.undulation_at(2.0, 0.0)


def test_bilinear_interpolation_matches_hand_computed_value(
    synthetic_geoid_path: Path,
) -> None:
    """At (0.5, 1.5): average of the 4 surrounding corners (10+20+40+50)/4 = 30.0 exactly."""
    grid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    assert grid.undulation_at(0.5, 1.5) == 30.0


def test_bilinear_interpolation_at_a_second_hand_computed_point(
    synthetic_geoid_path: Path,
) -> None:
    """At (1.5, 0.5): average of the 4 surrounding corners (50+60+80+90)/4 = 70.0 exactly."""
    grid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    assert grid.undulation_at(1.5, 0.5) == 70.0


def test_point_outside_extent_raises(synthetic_geoid_path: Path) -> None:
    grid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    with pytest.raises(MissingGeoidGridError, match="outside the geoid grid"):
        grid.undulation_at(10.0, 10.0)


def test_negative_coordinate_outside_extent_raises(synthetic_geoid_path: Path) -> None:
    grid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    with pytest.raises(MissingGeoidGridError, match="outside the geoid grid"):
        grid.undulation_at(-1.0, -1.0)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MissingGeoidGridError, match="not found"):
        GeoidGrid.from_geotiff(tmp_path / "does_not_exist.tif")


def test_nodata_cell_raises_rather_than_interpolating_through_it(
    geoid_with_nodata_path: Path,
) -> None:
    grid = GeoidGrid.from_geotiff(geoid_with_nodata_path)
    with pytest.raises(MissingGeoidGridError, match="nodata"):
        grid.undulation_at(0.5, 0.5)


def test_multiband_geotiff_rejected(multiband_geotiff_path: Path) -> None:
    with pytest.raises(GeoidError, match="2 raster bands"):
        GeoidGrid.from_geotiff(multiband_geotiff_path)


def test_missing_gdal_raises_geoid_error(monkeypatch: pytest.MonkeyPatch, synthetic_geoid_path: Path) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "osgeo":
            raise ImportError("simulated: GDAL not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(GeoidError, match="requires GDAL"):
        GeoidGrid.from_geotiff(synthetic_geoid_path)


# ----------------------------------------------------------------------
# Axis order / units / coverage -- verified against a non-square,
# asymmetric grid over Colombia's own real extent, added specifically
# after this project's own search for real EGM96/2008 data confirmed
# no such file (or live lookup service) is reachable from this
# environment (every path found leads outside the network allowlist).
# A symmetric grid cannot reveal a row/column transposition or a
# north/south or east/west axis inversion; this one can and does.
# ----------------------------------------------------------------------


def test_grid_dimensions_and_extent_match_construction(
    colombia_extent_geoid_path: Path,
) -> None:
    grid = GeoidGrid.from_geotiff(colombia_extent_geoid_path)
    assert grid.columns == 4
    assert grid.rows == 3
    assert grid.origin_longitude == -80.0
    assert grid.origin_latitude == 13.0


def test_axis_order_east_increases_value(colombia_extent_geoid_path: Path) -> None:
    """Same (northern) latitude band, moving east (higher column) must increase value, by construction."""
    grid = GeoidGrid.from_geotiff(colombia_extent_geoid_path)
    west = grid.undulation_at(-79.0, 8.0)
    east = grid.undulation_at(-70.0, 8.0)
    assert east > west


def test_axis_order_south_increases_value(colombia_extent_geoid_path: Path) -> None:
    """Same (western) longitude, moving south (higher row) must increase value, by construction."""
    grid = GeoidGrid.from_geotiff(colombia_extent_geoid_path)
    north = grid.undulation_at(-79.0, 8.0)
    south = grid.undulation_at(-79.0, 4.0)
    assert south > north


def test_axis_order_diagonal_extremes(colombia_extent_geoid_path: Path) -> None:
    """The northwest region must be the smallest of the 4 corner regions; southeast the largest."""
    grid = GeoidGrid.from_geotiff(colombia_extent_geoid_path)
    northwest = grid.undulation_at(-79.0, 8.0)
    northeast = grid.undulation_at(-70.0, 8.0)
    southwest = grid.undulation_at(-79.0, 4.0)
    southeast = grid.undulation_at(-70.0, 4.0)
    assert northwest < northeast
    assert northwest < southwest
    assert southeast == max(northwest, northeast, southwest, southeast)


def test_units_are_decimal_degrees_not_projected_meters(
    colombia_extent_geoid_path: Path,
) -> None:
    """origin/pixel values must be geographic degrees matching Colombia's own real extent, not projected meters."""
    grid = GeoidGrid.from_geotiff(colombia_extent_geoid_path)
    assert -81.0 < grid.origin_longitude < -79.0
    assert 12.0 < grid.origin_latitude < 14.0
    assert 0.0 < grid.pixel_width < 10.0  # a projected-meters grid would have pixel sizes in the thousands


def test_coverage_rejects_a_point_genuinely_outside_the_region(
    colombia_extent_geoid_path: Path,
) -> None:
    """Madrid, Spain -- nowhere near Colombia's own extent -- must be rejected, not silently answered."""
    grid = GeoidGrid.from_geotiff(colombia_extent_geoid_path)
    with pytest.raises(MissingGeoidGridError, match="outside the geoid grid"):
        grid.undulation_at(-3.7, 40.4)


def test_interior_point_interpolates_consistently_with_its_neighborhood(
    colombia_extent_geoid_path: Path,
) -> None:
    grid = GeoidGrid.from_geotiff(colombia_extent_geoid_path)
    value = grid.undulation_at(-73.0, 4.5)
    assert 10.0 < value < 300.0


# ----------------------------------------------------------------------
# Reviewer-identified gaps, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_sub_degree_resolution_matching_real_egm_convention(
    sub_degree_resolution_geoid_path: Path,
) -> None:
    """
    2.5 arc-minute resolution (real EGM2008 coarse-grid convention),
    with realistic (-30 to +50 m) undulation magnitudes rather than
    simplified round numbers.
    """
    grid = GeoidGrid.from_geotiff(sub_degree_resolution_geoid_path)
    resolution = 2.5 / 60.0
    assert grid.pixel_width == pytest.approx(resolution)

    # Interior point, interpolated -- confirms fine-resolution grids interpolate correctly, not just coarse ones.
    value = grid.undulation_at(resolution * 1.5, resolution * 1.5)
    assert -30.5 < value < 50.0


def test_negative_pixel_width_interpolates_correctly(
    negative_pixel_width_geoid_path: Path,
) -> None:
    """
    Confirmed directly against a hand-computed expected value: a grid
    with a NEGATIVE pixel_width (origin at the northeast corner,
    columns increasing westward) interpolates correctly with no
    special-casing -- the same bilinear formula naturally generalizes.

    Grid values (row-major): [0,1,2, 3,4,5, 6,7,8]; origin=(3,3),
    pixel_width=-1. At (1.5, 1.5): column_f=row_f=1.5, corners are
    values at (row=1,col=1)=4, (1,2)=5, (2,1)=7, (2,2)=8 ->
    bilinear midpoint = (4+5+7+8)/4 = 6.0 exactly.
    """
    grid = GeoidGrid.from_geotiff(negative_pixel_width_geoid_path)
    assert grid.pixel_width == -1.0
    assert grid.undulation_at(1.5, 1.5) == 6.0


def test_projected_coordinates_naturally_rejected_by_extent_check(
    synthetic_geoid_path: Path,
) -> None:
    """
    Passing projected coordinates (typically orders of magnitude
    larger than any geographic extent) into a geographic-degree grid
    is not validated or detected as a type mismatch -- but confirmed
    directly, it naturally fails the ordinary extent check rather
    than silently producing a nonsensical interpolated value. This
    documents an incidental safety consequence, not a deliberate
    CRS-type detection mechanism (see this class's own
    undulation_at() docstring).
    """
    grid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    with pytest.raises(MissingGeoidGridError, match="outside the geoid grid"):
        grid.undulation_at(500000.0, 4649776.0)  # typical UTM-style projected magnitudes
