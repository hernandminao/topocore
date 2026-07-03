from __future__ import annotations

import numpy as np
import pytest

from topocore.io.ply.enums import (
    PLYFormat,
    PLYScalarType,
    PLY_NUMPY_DTYPES,
)


# =============================================================================
# PLYFormat
# =============================================================================


class TestPLYFormat:
    """Tests for PLYFormat."""

    @pytest.mark.parametrize(
        ("member", "expected"),
        [
            (PLYFormat.ASCII, "ascii"),
            (
                PLYFormat.BINARY_LITTLE_ENDIAN,
                "binary_little_endian",
            ),
            (
                PLYFormat.BINARY_BIG_ENDIAN,
                "binary_big_endian",
            ),
        ],
    )
    def test_value(
        self,
        member: PLYFormat,
        expected: str,
    ) -> None:
        assert member.value == expected

    def test_is_string_enum(self) -> None:
        assert isinstance(PLYFormat.ASCII.value, str)

    @pytest.mark.parametrize(
        "value",
        [
            "ascii",
            "binary_little_endian",
            "binary_big_endian",
        ],
    )
    def test_lookup_from_string(
        self,
        value: str,
    ) -> None:
        member = PLYFormat(value)

        assert member.value == value

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "ASCII",
            "Binary",
            "little_endian",
            "json",
            "xml",
        ],
    )
    def test_invalid_lookup(
        self,
        value: str,
    ) -> None:
        with pytest.raises(ValueError):
            PLYFormat(value)

    def test_number_of_members(self) -> None:
        assert len(PLYFormat) == 3


# =============================================================================
# PLYScalarType
# =============================================================================


class TestPLYScalarType:
    """Tests for PLYScalarType."""

    @pytest.mark.parametrize(
        ("member", "dtype"),
        [
            (PLYScalarType.CHAR, np.dtype("i1")),
            (PLYScalarType.INT8, np.dtype("i1")),
            (PLYScalarType.UCHAR, np.dtype("u1")),
            (PLYScalarType.UINT8, np.dtype("u1")),
            (PLYScalarType.SHORT, np.dtype("i2")),
            (PLYScalarType.INT16, np.dtype("i2")),
            (PLYScalarType.USHORT, np.dtype("u2")),
            (PLYScalarType.UINT16, np.dtype("u2")),
            (PLYScalarType.INT, np.dtype("i4")),
            (PLYScalarType.INT32, np.dtype("i4")),
            (PLYScalarType.UINT, np.dtype("u4")),
            (PLYScalarType.UINT32, np.dtype("u4")),
            (PLYScalarType.FLOAT, np.dtype("f4")),
            (PLYScalarType.FLOAT32, np.dtype("f4")),
            (PLYScalarType.DOUBLE, np.dtype("f8")),
            (PLYScalarType.FLOAT64, np.dtype("f8")),
        ],
    )
    def test_numpy_dtype(
        self,
        member: PLYScalarType,
        dtype: np.dtype,
    ) -> None:
        assert member.numpy_dtype == dtype

    @pytest.mark.parametrize(
        "member",
        list(PLYScalarType),
    )
    def test_numpy_dtype_matches_mapping(
        self,
        member: PLYScalarType,
    ) -> None:
        assert member.numpy_dtype is PLY_NUMPY_DTYPES[member]

    @pytest.mark.parametrize(
        "member",
        list(PLYScalarType),
    )
    def test_mapping_contains_member(
        self,
        member: PLYScalarType,
    ) -> None:
        assert member in PLY_NUMPY_DTYPES

    @pytest.mark.parametrize(
        "member",
        list(PLYScalarType),
    )
    def test_mapping_values_are_numpy_dtype(
        self,
        member: PLYScalarType,
    ) -> None:
        assert isinstance(
            PLY_NUMPY_DTYPES[member],
            np.dtype,
        )

    @pytest.mark.parametrize(
        "value",
        [
            "char",
            "int8",
            "uchar",
            "uint8",
            "short",
            "int16",
            "ushort",
            "uint16",
            "int",
            "int32",
            "uint",
            "uint32",
            "float",
            "float32",
            "double",
            "float64",
        ],
    )
    def test_lookup_from_string(
        self,
        value: str,
    ) -> None:
        member = PLYScalarType(value)

        assert member.value == value

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "FLOAT",
            "Float",
            "integer",
            "bool",
            "string",
        ],
    )
    def test_invalid_lookup(
        self,
        value: str,
    ) -> None:
        with pytest.raises(ValueError):
            PLYScalarType(value)

    def test_all_scalar_types_are_mapped(self) -> None:
        assert set(PLY_NUMPY_DTYPES.keys()) == set(
            PLYScalarType,
        )

    def test_mapping_size_matches_enum(self) -> None:
        assert len(PLY_NUMPY_DTYPES) == len(
            PLYScalarType,
        )