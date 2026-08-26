"""Regression tests for topocore.io.ascii.type_inference."""

from __future__ import annotations

import numpy as np
import pytest

from topocore.io.ascii.type_inference import (
    COLUMN_DTYPES,
    TypeInferer,
)


@pytest.mark.parametrize(
    ("column_name", "expected_dtype", "values"),
    [
        ("x", np.float64, ["1.0", "2.5"]),
        ("y", np.float64, ["1.0", "2.5"]),
        ("z", np.float64, ["1.0", "2.5"]),
        ("easting", np.float64, ["1.0", "2.5"]),
        ("northing", np.float64, ["1.0", "2.5"]),
        ("elevation", np.float64, ["1.0", "2.5"]),
        ("height", np.float64, ["1.0", "2.5"]),
        ("gps_time", np.float64, ["1.0", "2.5"]),
        ("red", np.uint8, ["1", "255"]),
        ("green", np.uint8, ["1", "255"]),
        ("blue", np.uint8, ["1", "255"]),
        ("classification", np.uint8, ["1", "2"]),
        ("return_number", np.uint8, ["1", "2"]),
        ("number_of_returns", np.uint8, ["1", "2"]),
        ("intensity", np.int32, ["100", "200"]),
        ("point_source_id", np.int32, ["10", "20"]),
    ],
)
def test_known_column_dtype(
    column_name: str,
    expected_dtype: type[np.generic],
    values: list[str],
) -> None:
    result = TypeInferer.infer(
        column_name,
        values,
    )

    assert result.dtype == np.dtype(expected_dtype)


@pytest.mark.parametrize(
    "column_name",
    [
        "X",
        "Y",
        "Z",
        "EASTING",
        "Northing",
        "Elevation",
        "HEIGHT",
        "Gps_Time",
        "RED",
        "Green",
        "BLUE",
        "Classification",
        "RETURN_NUMBER",
        "Number_Of_Returns",
        "Intensity",
        "POINT_SOURCE_ID",
    ],
)
def test_known_column_names_are_case_insensitive(
    column_name: str,
) -> None:
    result = TypeInferer.infer(
        column_name,
        ["1", "2"],
    )

    expected_dtype = COLUMN_DTYPES[column_name.lower()]
    assert result.dtype == np.dtype(expected_dtype)


def test_unknown_column_infers_int32_first() -> None:
    result = TypeInferer.infer(
        "custom_attribute",
        ["1", "2", "-10", "100"],
    )

    assert result.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        result,
        np.array([1, 2, -10, 100], dtype=np.int32),
    )


def test_unknown_column_falls_back_to_float64() -> None:
    result = TypeInferer.infer(
        "custom_attribute",
        ["1.5", "2.25", "-3.75"],
    )

    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        result,
        np.array(
            [1.5, 2.25, -3.75],
            dtype=np.float64,
        ),
    )


def test_unknown_column_falls_back_to_string() -> None:
    result = TypeInferer.infer(
        "custom_attribute",
        ["ground", "building", "vegetation"],
    )

    assert result.dtype.kind in {"U", "S"}
    np.testing.assert_array_equal(
        result,
        np.array(
            ["ground", "building", "vegetation"],
            dtype=str,
        ),
    )


def test_unknown_column_with_mixed_values_falls_back_to_string() -> None:
    result = TypeInferer.infer(
        "custom_attribute",
        ["12", "unknown", "15.5"],
    )

    assert result.dtype.kind in {"U", "S"}

    np.testing.assert_array_equal(
        result,
        ["12", "unknown", "15.5"],
    )


def test_empty_values_for_known_float_column() -> None:
    result = TypeInferer.infer(
        "z",
        [],
    )

    assert result.dtype == np.dtype(np.float64)
    assert result.size == 0


def test_empty_values_for_unknown_column() -> None:
    result = TypeInferer.infer(
        "custom_attribute",
        [],
    )

    assert result.size == 0
    assert result.dtype == np.dtype(np.int32)
