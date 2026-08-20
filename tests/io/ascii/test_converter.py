"""
Regression suite for topocore.io.ascii.converter.ASCIIConverter --
PR19.

Includes a real bug found and fixed in this session:
ASCIIConverter is a standalone implementation (like LASConverter)
that does NOT go through BasePointConverter -- so the range-
validation fix applied there does not protect ASCII sources.
Confirmed directly: an "intensity" value of 70000 silently became
4464 via a plain .astype() cast to uint16. Also extends to
rejecting NaN/infinite values for integer-target columns (NumPy's
own float->int cast silently turns NaN into 0, not an exception).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.io.ascii.converter import ASCIIConverter
from topocore.io.ascii.records import ASCIIRecordBatch
from topocore.io.exceptions import CorruptedFileError
from topocore.pointcloud.attributes import PointAttribute


def _batch(**columns: np.ndarray) -> ASCIIRecordBatch:
    base = {"x": np.array([1.0]), "y": np.array([1.0]), "z": np.array([1.0])}
    base.update(columns)
    return ASCIIRecordBatch(columns=base)


def test_out_of_range_intensity_rejected_not_wrapped() -> None:
    """
    The exact regression: before the fix, this silently became 4464.
    """
    with pytest.raises(CorruptedFileError):
        ASCIIConverter.convert(_batch(intensity=np.array([70000.0])))


def test_in_range_intensity_still_works() -> None:
    chunk = ASCIIConverter.convert(_batch(intensity=np.array([5000.0])))
    assert chunk[PointAttribute.INTENSITY][0] == 5000


def test_nan_intensity_rejected() -> None:
    with pytest.raises(CorruptedFileError):
        ASCIIConverter.convert(_batch(intensity=np.array([np.nan])))


def test_color_within_uint16_range_accepted() -> None:
    """
    PointAttribute.COLOR's canonical dtype is uint16 (not uint8) --
    confirms 300 is legitimately valid, not a bug.
    """
    chunk = ASCIIConverter.convert(_batch(red=np.array([300]), green=np.array([0]), blue=np.array([0])))
    assert chunk[PointAttribute.COLOR][0, 0] == 300


def test_normal_combined_correctly() -> None:
    chunk = ASCIIConverter.convert(_batch(nx=np.array([0.0]), ny=np.array([0.0]), nz=np.array([1.0])))
    np.testing.assert_allclose(chunk[PointAttribute.NORMAL], [[0.0, 0.0, 1.0]])
