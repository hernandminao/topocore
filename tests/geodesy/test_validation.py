"""
Regression suite for topocore.geodesy.validation -- PR19.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geodesy.exceptions import ValidationError
from topocore.geodesy.validation import (
    validate_array,
    validate_bbox,
    validate_coordinate_arrays,
    validate_epsg,
    validate_lat_lon,
)

# ----------------------------------------------------------------------
# validate_lat_lon
# ----------------------------------------------------------------------


def test_valid_lat_lon_accepted() -> None:
    validate_lat_lon(40.3, -74.5)  # must not raise


@pytest.mark.parametrize("latitude", [90.0, -90.0])
def test_lat_lon_accepts_inclusive_latitude_boundaries(latitude: float) -> None:
    validate_lat_lon(latitude, 0.0)  # must not raise


@pytest.mark.parametrize("longitude", [180.0, -180.0])
def test_lat_lon_accepts_inclusive_longitude_boundaries(longitude: float) -> None:
    validate_lat_lon(0.0, longitude)  # must not raise


def test_lat_lon_rejects_latitude_above_90() -> None:
    with pytest.raises(ValidationError):
        validate_lat_lon(90.0001, 0.0)


def test_lat_lon_rejects_latitude_below_negative_90() -> None:
    with pytest.raises(ValidationError):
        validate_lat_lon(-90.0001, 0.0)


def test_lat_lon_rejects_longitude_above_180() -> None:
    with pytest.raises(ValidationError):
        validate_lat_lon(0.0, 180.0001)


def test_lat_lon_rejects_longitude_below_negative_180() -> None:
    with pytest.raises(ValidationError):
        validate_lat_lon(0.0, -180.0001)


def test_lat_lon_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        validate_lat_lon(float("nan"), 0.0)


def test_lat_lon_rejects_infinity() -> None:
    with pytest.raises(ValidationError):
        validate_lat_lon(0.0, float("inf"))


# ----------------------------------------------------------------------
# validate_epsg
# ----------------------------------------------------------------------


def test_valid_epsg_accepted() -> None:
    validate_epsg(4326)  # must not raise


def test_epsg_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        validate_epsg(0)


def test_epsg_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        validate_epsg(-4326)


def test_epsg_rejects_float() -> None:
    with pytest.raises(ValidationError):
        validate_epsg(4326.0)  # type: ignore[arg-type]


def test_epsg_rejects_string() -> None:
    with pytest.raises(ValidationError):
        validate_epsg("4326")  # type: ignore[arg-type]


def test_epsg_rejects_bool() -> None:
    """
    bool is a subclass of int in Python -- True/False could silently
    pass an `isinstance(x, int)` check and become EPSG:1/EPSG:0.
    Explicitly rejected, not a theoretical concern.
    """
    with pytest.raises(ValidationError):
        validate_epsg(True)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# validate_bbox
# ----------------------------------------------------------------------


def test_valid_bbox_accepted() -> None:
    validate_bbox((0.0, 0.0, 10.0, 10.0))  # must not raise


def test_bbox_rejects_wrong_length() -> None:
    with pytest.raises(ValidationError):
        validate_bbox((0.0, 0.0, 10.0))  # type: ignore[arg-type]


def test_bbox_rejects_minx_greater_than_maxx() -> None:
    with pytest.raises(ValidationError):
        validate_bbox((10.0, 0.0, 0.0, 10.0))


def test_bbox_rejects_miny_greater_than_maxy() -> None:
    with pytest.raises(ValidationError):
        validate_bbox((0.0, 10.0, 10.0, 0.0))


def test_bbox_accepts_zero_area_degenerate_box() -> None:
    """
    minx==maxx / miny==maxy (a single point as a bbox) is accepted --
    only minx > maxx / miny > maxy are rejected, not equality.
    """
    validate_bbox((5.0, 5.0, 5.0, 5.0))  # must not raise


def test_bbox_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        validate_bbox((0.0, 0.0, float("nan"), 10.0))


# ----------------------------------------------------------------------
# validate_array
# ----------------------------------------------------------------------


def test_valid_array_accepted_and_normalized_to_float64() -> None:
    result = validate_array(np.array([[1, 2], [3, 4]]), dims=2)
    assert result.dtype == np.float64
    assert result.shape == (2, 2)


def test_array_rejects_wrong_dimensionality() -> None:
    with pytest.raises(ValidationError):
        validate_array([1.0, 2.0, 3.0], dims=2)  # 1D, not 2D


def test_array_rejects_wrong_column_count() -> None:
    with pytest.raises(ValidationError):
        validate_array(np.array([[1, 2, 3], [4, 5, 6]]), dims=2)  # 3 columns, expected 2


def test_array_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        validate_array(np.array([[1.0, float("nan")], [3.0, 4.0]]), dims=2)


def test_array_rejects_infinity() -> None:
    with pytest.raises(ValidationError):
        validate_array(np.array([[1.0, float("inf")], [3.0, 4.0]]), dims=2)


# ----------------------------------------------------------------------
# validate_coordinate_arrays
# ----------------------------------------------------------------------


def test_valid_coordinate_arrays_accepted() -> None:
    x, y, z = validate_coordinate_arrays([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
    assert x.shape == y.shape == z.shape == (2,)


def test_coordinate_arrays_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        validate_coordinate_arrays([1.0, 2.0], [3.0, 4.0, 5.0])


def test_coordinate_arrays_rejects_no_arrays() -> None:
    with pytest.raises(ValidationError):
        validate_coordinate_arrays()


def test_coordinate_arrays_rejects_2d_input() -> None:
    with pytest.raises(ValidationError):
        validate_coordinate_arrays(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_coordinate_arrays_rejects_nan_in_any_array() -> None:
    with pytest.raises(ValidationError):
        validate_coordinate_arrays([1.0, float("nan")], [3.0, 4.0])
