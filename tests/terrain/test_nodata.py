"""
Regression suite for topocore.terrain.nodata -- PR19.
"""

from __future__ import annotations

import math

import numpy as np

from topocore.terrain.nodata import (
    fill_nodata,
    is_nodata,
    nodata_count,
    nodata_mask,
    replace_nodata,
    valid_count,
    valid_mask,
)


def test_is_nodata_true_for_nan() -> None:
    assert is_nodata(float("nan")) is True


def test_is_nodata_true_for_infinity() -> None:
    assert is_nodata(float("inf")) is True
    assert is_nodata(float("-inf")) is True


def test_is_nodata_false_for_finite_value() -> None:
    assert is_nodata(42.0) is False
    assert is_nodata(0.0) is False
    assert is_nodata(-100.5) is False


def test_valid_mask_and_nodata_mask_are_complementary() -> None:
    array = np.array([1.0, float("nan"), 3.0, float("inf")])
    valid = valid_mask(array)
    nodata = nodata_mask(array)

    np.testing.assert_array_equal(valid, [True, False, True, False])
    np.testing.assert_array_equal(nodata, [False, True, False, True])
    np.testing.assert_array_equal(valid, ~nodata)


def test_valid_count_and_nodata_count_sum_to_total() -> None:
    array = np.array([1.0, float("nan"), 3.0, float("inf"), 5.0])
    assert valid_count(array) + nodata_count(array) == array.size
    assert valid_count(array) == 3
    assert nodata_count(array) == 2


def test_replace_nodata_only_touches_nodata_cells() -> None:
    array = np.array([1.0, float("nan"), 3.0])
    result = replace_nodata(array, -9999.0)

    np.testing.assert_array_equal(result, [1.0, -9999.0, 3.0])


def test_replace_nodata_does_not_mutate_original() -> None:
    array = np.array([1.0, float("nan"), 3.0])
    replace_nodata(array, -9999.0)
    assert math.isnan(array[1])  # original untouched


def test_fill_nodata_defaults_to_zero() -> None:
    array = np.array([1.0, float("nan"), 3.0])
    result = fill_nodata(array)
    np.testing.assert_array_equal(result, [1.0, 0.0, 3.0])


def test_fill_nodata_is_replace_nodata_with_given_value() -> None:
    array = np.array([1.0, float("nan"), 3.0])
    np.testing.assert_array_equal(fill_nodata(array, -1.0), replace_nodata(array, -1.0))
