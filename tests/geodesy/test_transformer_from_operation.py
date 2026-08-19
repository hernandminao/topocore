"""
Regression suite for CoordinateTransformer.from_operation() -- part
of PR19 (QA/Validation), module Geodesy. Rebuilt from scratch: the
real repository currently has 0 tests (confirmed with Hernán), even
though the module's own docstrings describe extensive prior
verification (EPSG formula 9606, hand-derivable geometry at
lon=0/lat=0). That verification is redone here, empirically, against
real pyproj/PROJ behavior -- not assumed from the docstring's claims.

Independent verification note (found while writing this suite, not
before): a first attempt at an independent EPSG 9606 cross-check
used the WRONG sign for the rotation terms (a classic Position
Vector vs. Coordinate Frame Rotation convention mix-up), and
disagreed with the real code's output by tens of meters. Isolating
PROJ's own `+proj=helmert +convention=position_vector` step (bypassing
geographic<->geocentric conversion entirely) and testing each of the
7 parameters individually confirmed: translations and scale matched
immediately; only the three rotation terms disagreed, and all three
in the same "opposite sign" direction. This pinpointed the bug to
the verification formula itself, not topocore.geodesy.transformer,
which never re-implements the rotation matrix -- it only builds a
PROJ pipeline string and lets PROJ execute it. The corrected sign
convention (confirmed to match PROJ exactly, 0.0 diff at double
precision) is what `_epsg_9606_position_vector` below implements.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pyproj import Transformer as PyprojTransformer

from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import TransformationError, ValidationError
from topocore.geodesy.grid_shift import GridShift, GridShiftFormat
from topocore.geodesy.helmert import HelmertParameters
from topocore.geodesy.operation import CoordinateOperation
from topocore.geodesy.operation_type import OperationType
from topocore.geodesy.transformer import CoordinateTransformer

_ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


def _epsg_9606_position_vector(
    x: float,
    y: float,
    z: float,
    helmert: HelmertParameters,
) -> tuple[float, float, float]:
    """
    Independent (not calling any topocore code) implementation of
    the EPSG 9606 Position Vector 7-parameter Helmert formula,
    confirmed empirically to match PROJ's own
    `+proj=helmert +convention=position_vector` to double-precision
    exactness -- see module docstring for how the sign convention
    was pinned down.
    """
    rx = helmert.rx * _ARCSEC_TO_RAD
    ry = helmert.ry * _ARCSEC_TO_RAD
    rz = helmert.rz * _ARCSEC_TO_RAD
    ds = helmert.scale * 1e-6

    x2 = helmert.tx + (1 + ds) * (x - rz * y + ry * z)
    y2 = helmert.ty + (1 + ds) * (rz * x + y - rx * z)
    z2 = helmert.tz + (1 + ds) * (-ry * x + rx * y + z)

    return x2, y2, z2


def _geographic_to_geocentric(crs: CRS, lon: float, lat: float, height: float) -> tuple[float, float, float]:
    ellipsoid = crs.ellipsoid
    assert ellipsoid is not None

    pipeline = PyprojTransformer.from_pipeline(
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
        f"+step +proj=cart +a={ellipsoid.semi_major_axis} +rf={ellipsoid.inverse_flattening}"
    )
    return pipeline.transform(lon, lat, height)


def _geocentric_to_geographic(crs: CRS, x: float, y: float, z: float) -> tuple[float, float, float]:
    ellipsoid = crs.ellipsoid
    assert ellipsoid is not None

    pipeline = PyprojTransformer.from_pipeline(
        f"+proj=pipeline +step +inv +proj=cart +a={ellipsoid.semi_major_axis} +rf={ellipsoid.inverse_flattening} "
        "+step +proj=unitconvert +xy_in=rad +xy_out=deg"
    )
    return pipeline.transform(x, y, z)


# ----------------------------------------------------------------------
# IDENTITY
# ----------------------------------------------------------------------


def test_identity_delegates_to_plain_constructor() -> None:
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(4269)
    operation = CoordinateOperation(source_crs=source, target_crs=target, operation_type=OperationType.IDENTITY)

    from_operation = CoordinateTransformer.from_operation(operation)
    plain = CoordinateTransformer(source, target)

    point = (-74.5, 40.3, 150.0)
    assert from_operation.transform_point(*point) == plain.transform_point(*point)
    assert from_operation.source_crs == source
    assert from_operation.target_crs == target


# ----------------------------------------------------------------------
# HELMERT -- isolated, hand-derivable geometry at (lon=0, lat=0)
# ----------------------------------------------------------------------


def test_pure_translation_shifts_only_height_at_equator_prime_meridian() -> None:
    """
    At (lon=0, lat=0), geocentric X is the local vertical. A pure
    tx=100m translation should shift height by ~exactly 100m and
    leave lon/lat unchanged at this specific point.
    """
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(4269)
    helmert = HelmertParameters(tx=100.0, ty=0.0, tz=0.0)
    operation = CoordinateOperation(
        source_crs=source,
        target_crs=target,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )

    transformer = CoordinateTransformer.from_operation(operation)
    lon, lat, height = transformer.transform_point(0.0, 0.0, 0.0)

    assert lon == pytest.approx(0.0, abs=1e-9)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert height == pytest.approx(100.0, abs=1e-6)


def test_pure_rz_rotation_shifts_only_longitude_at_equator_prime_meridian() -> None:
    """
    At (lon=0, lat=0), a pure rz rotation directly rotates the
    equatorial angle -- longitude should shift by ~exactly the
    rotation angle (in the same units, arcsec here).
    """
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(4269)
    helmert = HelmertParameters(tx=0.0, ty=0.0, tz=0.0, rz=1.0)
    operation = CoordinateOperation(
        source_crs=source,
        target_crs=target,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )

    transformer = CoordinateTransformer.from_operation(operation)
    lon, lat, _ = transformer.transform_point(0.0, 0.0, 0.0)

    assert lon * 3600.0 == pytest.approx(1.0, abs=1e-6)
    assert lat == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# HELMERT -- general (non-degenerate) point, all 7 parameters at once,
# verified against an independent EPSG 9606 implementation.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "height", "tx", "ty", "tz", "rx", "ry", "rz", "scale"),
    [
        (-74.5, 40.3, 150.0, 50.0, -30.0, 20.0, 2.0, -1.5, 0.8, 5.0),
        (139.7, 35.7, 20.0, -10.0, 15.0, -5.0, -0.5, 0.3, -0.2, -1.0),
        (0.0, -33.9, 500.0, 100.0, 100.0, 100.0, 3.0, 3.0, 3.0, 10.0),
    ],
)
def test_general_helmert_matches_independent_epsg_9606_implementation(
    lon: float,
    lat: float,
    height: float,
    tx: float,
    ty: float,
    tz: float,
    rx: float,
    ry: float,
    rz: float,
    scale: float,
) -> None:
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(4269)
    helmert = HelmertParameters(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz, scale=scale)
    operation = CoordinateOperation(
        source_crs=source,
        target_crs=target,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )

    transformer = CoordinateTransformer.from_operation(operation)
    result_lon, result_lat, result_height = transformer.transform_point(lon, lat, height)

    x, y, z = _geographic_to_geocentric(source, lon, lat, height)
    x2, y2, z2 = _epsg_9606_position_vector(x, y, z, helmert)
    expected_lon, expected_lat, expected_height = _geocentric_to_geographic(target, x2, y2, z2)

    assert result_lon == pytest.approx(expected_lon, abs=1e-9)
    assert result_lat == pytest.approx(expected_lat, abs=1e-9)
    assert result_height == pytest.approx(expected_height, abs=1e-6)


# ----------------------------------------------------------------------
# Explicitly unsupported cases -- one regression test per row of the
# CoordinateTransformer.from_operation() docstring's table.
# ----------------------------------------------------------------------


def test_grid_shift_raises_not_implemented() -> None:
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(4269)
    grid_shift = GridShift(path=Path(__file__), format=GridShiftFormat.NTV2)  # path existence never validated
    operation = CoordinateOperation(
        source_crs=source,
        target_crs=target,
        operation_type=OperationType.GRID_SHIFT,
        grid_shift=grid_shift,
    )

    with pytest.raises(TransformationError, match="GRID_SHIFT"):
        CoordinateTransformer.from_operation(operation)


def test_time_dependent_helmert_raises_not_implemented() -> None:
    source = CRS.from_epsg(4326)
    helmert = HelmertParameters(tx=1.0, ty=1.0, tz=1.0, dtx=0.01, reference_epoch=2018.0)
    operation = CoordinateOperation(
        source_crs=source,
        target_crs=source,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )

    with pytest.raises(TransformationError, match="time-dependent"):
        CoordinateTransformer.from_operation(operation)


def test_helmert_between_projected_crs_raises() -> None:
    source = CRS.from_epsg(32617)  # UTM 17N -- projected, not geographic
    target = CRS.from_epsg(32618)  # UTM 18N
    helmert = HelmertParameters(tx=1.0, ty=1.0, tz=1.0)
    operation = CoordinateOperation(
        source_crs=source,
        target_crs=target,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )

    with pytest.raises(TransformationError, match="geographic"):
        CoordinateTransformer.from_operation(operation)


def test_helmert_source_projected_target_geographic_raises() -> None:
    source = CRS.from_epsg(32617)  # projected
    target = CRS.from_epsg(4269)  # geographic
    helmert = HelmertParameters(tx=1.0, ty=1.0, tz=1.0)
    operation = CoordinateOperation(
        source_crs=source,
        target_crs=target,
        operation_type=OperationType.HELMERT,
        helmert=helmert,
    )

    with pytest.raises(TransformationError, match="geographic"):
        CoordinateTransformer.from_operation(operation)


# ----------------------------------------------------------------------
# CoordinateOperation's own construction-time validation (consumed by
# from_operation(), so its guarantees matter here too).
# ----------------------------------------------------------------------


def test_helmert_operation_without_helmert_parameters_rejected_at_construction() -> None:
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(4269)

    with pytest.raises(ValidationError):
        CoordinateOperation(source_crs=source, target_crs=target, operation_type=OperationType.HELMERT)


def test_identity_operation_with_helmert_parameters_rejected_at_construction() -> None:
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(4269)
    helmert = HelmertParameters(tx=1.0, ty=1.0, tz=1.0)

    with pytest.raises(ValidationError):
        CoordinateOperation(
            source_crs=source,
            target_crs=target,
            operation_type=OperationType.IDENTITY,
            helmert=helmert,
        )
