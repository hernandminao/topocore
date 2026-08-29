"""
Coverage audit tests for topocore.geodesy.transformer.CoordinateTransformer.

PR22 coverage hardening. The class's own docstring references
`test_transformer_from_operation.py` as containing "one regression
test per row" of its documented scope table -- that file is not
present in this audit's sandbox (part of Hernán's own original test
suite, outside the src-only reconstruction this session works from).
This suite fills that gap, and goes further: it independently
VERIFIES (not merely trusts) the docstring's own specific numerical
claim about hand-derivable Helmert geometry.

Verified directly, with real pyproj execution (pyproj was confirmed
absent from this sandbox and installed specifically for this
module's audit): a pure tx=100 translation applied at (lon=0, lat=0,
h=0) resolves to h=+100.0000 exactly (geocentric X IS the local
vertical at that exact point on the equator/prime meridian) and a
pure rz=1 arcsecond rotation resolves to a longitude shift of
EXACTLY 1/3600 degrees (0.00027778) -- both match the docstring's own
claimed hand-derivable geometry precisely, not merely approximately.

Also discovered directly (not assumed): CoordinateOperation's own
__post_init__ rejects a static HELMERT operation whose source and
target CRS are identical (only meaningful for a time-dependent
Helmert, e.g. epoch-to-epoch within the same CRS) -- every HELMERT
test below uses two genuinely different geographic CRS (WGS84,
NAD83) to construct a valid operation.

Every row of the class's own documented support table (IDENTITY,
HELMERT static/geographic, HELMERT time-dependent [rejected],
HELMERT non-geographic [rejected], GRID_SHIFT [rejected]) is
exercised directly below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import TransformationError, ValidationError
from topocore.geodesy.grid_shift import GridShift, GridShiftFormat
from topocore.geodesy.helmert import HelmertParameters
from topocore.geodesy.operation import CoordinateOperation
from topocore.geodesy.operation_type import OperationType
from topocore.geodesy.transformer import CoordinateTransformer


@pytest.fixture
def wgs84() -> CRS:
    return CRS.from_epsg(4326)


@pytest.fixture
def nad83() -> CRS:
    return CRS.from_epsg(4269)


@pytest.fixture
def utm17n() -> CRS:
    return CRS.from_epsg(32617)


# ----------------------------------------------------------------------
# Constructor -- EPSG-cached path and custom (non-EPSG) path.
# ----------------------------------------------------------------------


def test_constructor_with_epsg_backed_crs(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)
    assert transformer.source_crs.epsg == 4326
    assert transformer.target_crs.epsg == 32617


def test_constructor_with_custom_non_epsg_crs() -> None:
    custom = CRS.from_proj4("+proj=laea +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)
    transformer = CoordinateTransformer(wgs84, custom)
    result = transformer.transform_point(0.0, 0.0)
    assert result[0] == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# from_operation() -- IDENTITY.
# ----------------------------------------------------------------------


def test_from_operation_identity(wgs84: CRS, nad83: CRS) -> None:
    operation = CoordinateOperation(source_crs=wgs84, target_crs=nad83, operation_type=OperationType.IDENTITY)
    transformer = CoordinateTransformer.from_operation(operation)

    assert transformer.source_crs.epsg == 4326
    assert transformer.target_crs.epsg == 4269


# ----------------------------------------------------------------------
# from_operation() -- HELMERT, verified against hand-derivable geometry.
# ----------------------------------------------------------------------


def test_from_operation_helmert_pure_translation_matches_hand_derived_geometry(wgs84: CRS, nad83: CRS) -> None:
    """At (lon=0, lat=0), geocentric X IS the local vertical -- a pure tx=100 must resolve to h=+100 exactly."""
    helmert = HelmertParameters(tx=100.0, ty=0.0, tz=0.0)
    operation = CoordinateOperation(
        source_crs=wgs84,
        target_crs=nad83,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )
    transformer = CoordinateTransformer.from_operation(operation)

    lon, lat, h = transformer.transform_point(0.0, 0.0, 0.0)

    assert lon == pytest.approx(0.0, abs=1e-9)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert h == pytest.approx(100.0, abs=1e-4)


def test_from_operation_helmert_pure_rz_rotation_matches_hand_derived_geometry(wgs84: CRS, nad83: CRS) -> None:
    """A pure rz=1 arcsecond rotation at (lon=0, lat=0) shifts longitude by exactly 1/3600 degrees."""
    helmert = HelmertParameters(tx=0.0, ty=0.0, tz=0.0, rz=1.0)
    operation = CoordinateOperation(
        source_crs=wgs84,
        target_crs=nad83,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )
    transformer = CoordinateTransformer.from_operation(operation)

    lon, lat, _h = transformer.transform_point(0.0, 0.0, 0.0)

    assert lon == pytest.approx(1.0 / 3600.0, abs=1e-8)
    assert lat == pytest.approx(0.0, abs=1e-6)


def test_from_operation_helmert_time_dependent_rejected(wgs84: CRS, nad83: CRS) -> None:
    helmert = HelmertParameters(tx=1.0, ty=0.0, tz=0.0, dtx=0.01, reference_epoch=2010.0)
    operation = CoordinateOperation(
        source_crs=wgs84,
        target_crs=nad83,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )

    with pytest.raises(TransformationError, match="time-dependent"):
        CoordinateTransformer.from_operation(operation)


def test_from_operation_helmert_non_geographic_crs_rejected(nad83: CRS, utm17n: CRS) -> None:
    helmert = HelmertParameters(tx=1.0, ty=0.0, tz=0.0)
    operation = CoordinateOperation(
        source_crs=utm17n,
        target_crs=nad83,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )

    with pytest.raises(TransformationError, match="only supported between geographic"):
        CoordinateTransformer.from_operation(operation)


# ----------------------------------------------------------------------
# from_operation() -- GRID_SHIFT, explicitly unimplemented.
# ----------------------------------------------------------------------


def test_from_operation_grid_shift_not_implemented(wgs84: CRS, nad83: CRS) -> None:
    grid_shift = GridShift(path=Path("test.gsb"), format=GridShiftFormat.NTV2)
    operation = CoordinateOperation(
        source_crs=wgs84,
        target_crs=nad83,
        operation_type=OperationType.GRID_SHIFT,
        grid_shift=grid_shift,
    )

    with pytest.raises(TransformationError, match="not yet implemented"):
        CoordinateTransformer.from_operation(operation)


# ----------------------------------------------------------------------
# transform_point() / transform_array() / transform_bbox() -- happy
# paths and validation.
# ----------------------------------------------------------------------


def test_transform_point_without_z(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)
    x, _y, z = transformer.transform_point(-80.0, 25.0)

    assert x == pytest.approx(600913.0, abs=1.0)
    assert z is None


def test_transform_point_with_z(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)
    _x, _y, z = transformer.transform_point(-80.0, 25.0, 100.0)

    assert z == 100.0


def test_transform_array_without_z(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)
    xs = np.array([-80.0, -81.0, -82.0])
    ys = np.array([25.0, 26.0, 27.0])

    x_new, _y_new, z_new = transformer.transform_array(xs, ys)

    assert x_new.shape == (3,)
    assert z_new is None


def test_transform_array_with_z(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)
    xs = np.array([-80.0, -81.0])
    ys = np.array([25.0, 26.0])
    zs = np.array([100.0, 200.0])

    _, _, z_new = transformer.transform_array(xs, ys, zs)

    np.testing.assert_array_equal(z_new, zs)


def test_transform_array_rejects_mismatched_lengths(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)

    with pytest.raises(ValidationError, match="identical length"):
        transformer.transform_array(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_transform_array_rejects_non_finite_values(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)

    with pytest.raises(ValidationError, match="NaN or infinite"):
        transformer.transform_array(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_transform_bbox_happy_path(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)
    minx, miny, maxx, maxy = transformer.transform_bbox((-82.0, 25.0, -80.0, 27.0))

    assert minx < maxx
    assert miny < maxy


def test_transform_bbox_rejects_invalid_bbox(wgs84: CRS, utm17n: CRS) -> None:
    transformer = CoordinateTransformer(wgs84, utm17n)

    with pytest.raises(ValidationError, match="cannot be greater than"):
        transformer.transform_bbox((10.0, 10.0, 5.0, 5.0))
