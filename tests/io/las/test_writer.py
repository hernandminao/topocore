"""
Regression suite for topocore.io.las.writer.LASWriter -- PR19.

Includes a real bug found and fixed in this session: LASWriter never
set header.scales/header.offsets before writing, silently relying on
laspy's own internal default (1cm scale). Confirmed directly with a
write->read round trip on realistic UTM-style survey coordinates:
500123.456 became 500123.46 -- a real precision loss for GNSS
RTK-grade (millimeter-precision) survey workflows, which this
library is explicitly built around.

Fixed by defaulting to a 1mm scale (the ASPRS-recommended default,
finer than laspy's own) when not explicitly configured, and
auto-computing an offset from the actual data's minimum coordinate
per axis -- while still allowing the caller to pass explicit
scale/offset for even finer control.
"""

from __future__ import annotations

import laspy  # type: ignore[import-untyped]
import numpy as np
import pytest

from topocore.io.las.reader import LASReader
from topocore.io.las.writer import LASWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud


def _make_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=len(x), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = x
    chunk[PointAttribute.Y][:] = y
    chunk[PointAttribute.Z][:] = z
    cloud.add_chunk(chunk)
    return cloud


def test_default_scale_preserves_millimeter_precision(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    The exact regression: before the fix, this round trip lost
    precision beyond 1cm (500123.456 -> 500123.46). Now preserves
    1mm precision by default.
    """
    path = str(tmp_path / "utm.las")
    known_x = np.array([500123.456, 500124.789, 500125.111])
    known_y = np.array([4000456.222, 4000457.333, 4000458.444])
    known_z = np.array([1250.500, 1251.750, 1249.250])

    LASWriter(path).write(_make_cloud(known_x, known_y, known_z))

    with laspy.open(path) as f:
        assert list(f.header.scales) == pytest.approx([0.001, 0.001, 0.001])

    with LASReader(path) as reader:
        result = next(iter(reader.read()))

    np.testing.assert_allclose(result[PointAttribute.X], known_x, atol=1e-3)
    np.testing.assert_allclose(result[PointAttribute.Y], known_y, atol=1e-3)
    np.testing.assert_allclose(result[PointAttribute.Z], known_z, atol=1e-3)


def test_explicit_scale_and_offset_are_respected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "explicit.las")
    cloud = _make_cloud(np.array([500000.1234]), np.array([4000000.5678]), np.array([100.0]))

    LASWriter(path, scale=(0.0001, 0.0001, 0.0001), offset=(500000.0, 4000000.0, 0.0)).write(cloud)

    with laspy.open(path) as f:
        assert list(f.header.scales) == pytest.approx([0.0001, 0.0001, 0.0001])
        assert list(f.header.offsets) == pytest.approx([500000.0, 4000000.0, 0.0])


def test_offset_auto_computed_from_data_minimum(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "auto_offset.las")
    x = np.array([500100.0, 500200.0, 500300.0])
    y = np.array([4000100.0, 4000200.0, 4000300.0])
    z = np.array([100.0, 200.0, 300.0])

    LASWriter(path).write(_make_cloud(x, y, z))

    with laspy.open(path) as f:
        assert f.header.offsets[0] == pytest.approx(500100.0)
        assert f.header.offsets[1] == pytest.approx(4000100.0)
        assert f.header.offsets[2] == pytest.approx(100.0)


def test_round_trip_color_and_scalar_attributes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "roundtrip.las")
    cloud = PointCloud()
    chunk = Chunk(
        size=2,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.INTENSITY,
            PointAttribute.COLOR,
        ],
    )
    chunk[PointAttribute.X][:] = [1.0, 2.0]
    chunk[PointAttribute.Y][:] = [1.0, 2.0]
    chunk[PointAttribute.Z][:] = [1.0, 2.0]
    chunk[PointAttribute.INTENSITY][:] = [100, 200]
    chunk[PointAttribute.COLOR][:] = [[255, 0, 0], [0, 255, 0]]
    cloud.add_chunk(chunk)

    LASWriter(path).write(cloud)

    with LASReader(path) as reader:
        result = next(iter(reader.read()))

    np.testing.assert_array_equal(result[PointAttribute.INTENSITY], [100, 200])
    np.testing.assert_array_equal(result[PointAttribute.COLOR], [[255, 0, 0], [0, 255, 0]])
