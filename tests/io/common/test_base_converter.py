"""
Regression suite for topocore.io.common.base_converter -- PR19.

Includes a real, severe bug found and fixed in this session: casting
a source array to a narrower integer dtype (e.g. int32 -> uint16, the
exact case for "intensity") via a plain .astype() call does NOT
raise in NumPy -- it silently wraps out-of-range integers modulo the
target type's range. Confirmed directly (via both ASCII and PLY
sources, both routing through this shared method) that a value of
70000 silently became 4464 (70000 % 65536), with no error or
warning anywhere. Fixed by validating the actual value range against
the target dtype's representable range BEFORE casting.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.io.common.base_converter import BasePointConverter
from topocore.io.common.records import PointRecordBatch
from topocore.io.exceptions import CorruptedFileError
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk


class _DummyConverter(BasePointConverter):
    @property
    def attribute_mapping(self) -> dict[str, PointAttribute]:
        return {}

    def _populate_special_attributes(self, chunk: Chunk, batch: PointRecordBatch) -> None:
        pass


def test_out_of_range_intensity_rejected_not_wrapped() -> None:
    """
    The exact regression: before the fix, this silently became 4464
    (70000 % 65536) instead of raising.
    """
    batch = PointRecordBatch(
        arrays={
            "x": np.array([1.0]),
            "y": np.array([1.0]),
            "z": np.array([1.0]),
            "intensity": np.array([70000.0]),
        }
    )

    with pytest.raises(CorruptedFileError):
        _DummyConverter().convert(batch)


def test_in_range_intensity_still_works() -> None:
    batch = PointRecordBatch(
        arrays={
            "x": np.array([1.0]),
            "y": np.array([1.0]),
            "z": np.array([1.0]),
            "intensity": np.array([5000.0]),
        }
    )
    chunk = _DummyConverter().convert(batch)
    assert chunk[PointAttribute.INTENSITY][0] == 5000


def test_negative_value_rejected_for_unsigned_target() -> None:
    batch = PointRecordBatch(
        arrays={
            "x": np.array([1.0]),
            "y": np.array([1.0]),
            "z": np.array([1.0]),
            "intensity": np.array([-1.0]),
        }
    )
    with pytest.raises(CorruptedFileError):
        _DummyConverter().convert(batch)


def test_nan_in_float_source_for_integer_attribute_is_rejected() -> None:
    """
    Extends the same "corrupt data must fail loud" principle: NumPy's
    own float->int cast silently turns NaN into 0 with only a
    RuntimeWarning, not an exception -- confirmed directly before
    fixing this. A NaN in an integer-target column like
    "classification" (unlike elevation, which legitimately uses NaN
    for NoData) indicates a genuinely malformed source file.
    """
    batch = PointRecordBatch(
        arrays={
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([1.0, 2.0, 3.0]),
            "z": np.array([1.0, 2.0, 3.0]),
            "classification": np.array([2.0, np.nan, 5.0]),
        }
    )
    with pytest.raises(CorruptedFileError):
        _DummyConverter().convert(batch)


def test_finite_float_values_for_integer_attribute_still_work() -> None:
    batch = PointRecordBatch(
        arrays={
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([1.0, 2.0, 3.0]),
            "z": np.array([1.0, 2.0, 3.0]),
            "classification": np.array([2.0, 3.0, 5.0]),
        }
    )
    chunk = _DummyConverter().convert(batch)
    np.testing.assert_array_equal(chunk[PointAttribute.CLASSIFICATION], [2, 3, 5])
