"""
Regression suite for the small geodesy value objects -- PR19:
TransformationAccuracy, LocalCRS, VerticalDatum, Datum, ProjectionInfo,
Ellipsoid, and the internal _cache module.
"""

from __future__ import annotations

import pytest

from topocore.geodesy import _cache
from topocore.geodesy.accuracy import TransformationAccuracy
from topocore.geodesy.crs import CRS
from topocore.geodesy.datum import Datum
from topocore.geodesy.ellipsoid import Ellipsoid
from topocore.geodesy.exceptions import ValidationError
from topocore.geodesy.local_crs import LocalCRS
from topocore.geodesy.projection import ProjectionInfo
from topocore.geodesy.vertical_datum import VerticalDatum

# ----------------------------------------------------------------------
# TransformationAccuracy
# ----------------------------------------------------------------------


def test_transformation_accuracy_valid() -> None:
    accuracy = TransformationAccuracy(value=0.05, unit="m", source="EPSG registry")
    assert accuracy.value == 0.05
    assert accuracy.estimated is False


def test_transformation_accuracy_rejects_negative_value() -> None:
    with pytest.raises(ValidationError):
        TransformationAccuracy(value=-0.01)


def test_transformation_accuracy_zero_is_valid() -> None:
    TransformationAccuracy(value=0.0)  # must not raise -- zero is not negative


def test_transformation_accuracy_default_unit_is_meters() -> None:
    accuracy = TransformationAccuracy(value=1.0)
    assert accuracy.unit == "m"


# ----------------------------------------------------------------------
# LocalCRS
# ----------------------------------------------------------------------


def test_local_crs_valid() -> None:
    base = CRS.from_epsg(32617)
    local = LocalCRS(name="Proyecto Norte", base_crs=base, origin_x=500000.0, origin_y=4500000.0)
    assert local.rotation == 0.0
    assert local.scale == 1.0


def test_local_crs_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        LocalCRS(name="", base_crs=CRS.from_epsg(32617), origin_x=0.0, origin_y=0.0)


def test_local_crs_rejects_zero_scale() -> None:
    with pytest.raises(ValidationError):
        LocalCRS(
            name="A",
            base_crs=CRS.from_epsg(32617),
            origin_x=0.0,
            origin_y=0.0,
            scale=0.0,
        )


def test_local_crs_rejects_negative_scale() -> None:
    with pytest.raises(ValidationError):
        LocalCRS(
            name="A",
            base_crs=CRS.from_epsg(32617),
            origin_x=0.0,
            origin_y=0.0,
            scale=-1.0,
        )


# ----------------------------------------------------------------------
# VerticalDatum
# ----------------------------------------------------------------------


def test_vertical_datum_valid() -> None:
    datum = VerticalDatum(name="EGM96", authority="NGA")
    assert datum.name == "EGM96"


def test_vertical_datum_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        VerticalDatum(name="")


# ----------------------------------------------------------------------
# Datum / ProjectionInfo / Ellipsoid -- plain data, no validation
# ----------------------------------------------------------------------


def test_datum_construction() -> None:
    datum = Datum(name="World Geodetic System 1984")
    assert datum.remarks is None
    assert datum.scope is None


def test_projection_info_construction() -> None:
    info = ProjectionInfo(name="UTM zone 17N", method_name="Transverse Mercator")
    assert info.accuracy is None


def test_ellipsoid_construction() -> None:
    ellipsoid = Ellipsoid(
        name="WGS 84",
        semi_major_axis=6378137.0,
        semi_minor_axis=6356752.314245179,
        inverse_flattening=298.257223563,
        is_semi_minor_computed=True,
    )
    assert ellipsoid.semi_major_axis == pytest.approx(6378137.0)


# ----------------------------------------------------------------------
# _cache
# ----------------------------------------------------------------------


def test_get_crs_returns_same_object_on_repeated_calls() -> None:
    _cache.clear_cache()
    first = _cache.get_crs(4326)
    second = _cache.get_crs(4326)
    assert first is second  # lru_cache identity, not just equality


def test_get_transformer_returns_same_object_on_repeated_calls() -> None:
    _cache.clear_cache()
    first = _cache.get_transformer(32617, 4326)
    second = _cache.get_transformer(32617, 4326)
    assert first is second


def test_get_geod_returns_same_object_on_repeated_calls() -> None:
    _cache.clear_cache()
    first = _cache.get_geod(6378137.0, 298.257223563)
    second = _cache.get_geod(6378137.0, 298.257223563)
    assert first is second


def test_clear_cache_actually_clears() -> None:
    _cache.get_crs(4326)
    info_before = _cache.get_crs.cache_info()
    assert info_before.currsize > 0

    _cache.clear_cache()
    info_after = _cache.get_crs.cache_info()
    assert info_after.currsize == 0


def test_cache_info_reports_all_three_caches() -> None:
    _cache.clear_cache()
    info = _cache.cache_info()
    assert set(info.keys()) == {"crs", "transformer", "geod"}
