"""
Regression suite for topocore.io.laz.writer.LAZWriter -- PR19.

Includes a real bug found and fixed in this session: LAZWriter is a
near-complete duplicate of LASWriter, sharing the exact same bug --
it never set header.scales/header.offsets, silently relying on
laspy's coarser 1cm default. Confirmed directly with a write->read
round trip through a REAL compressed .laz file (lazrs backend, not
a mock): 500123.456 became 500123.46. Fixed identically to
LASWriter -- 1mm default scale, auto-computed offset, explicit
override supported.
"""

from __future__ import annotations

import laspy  # type: ignore[import-untyped]
import numpy as np
import pytest

from topocore.io.laz.reader import LAZReader
from topocore.io.laz.writer import LAZWriter
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
    The exact regression, reproduced through a real compressed .laz
    file: before the fix, this round trip lost precision beyond 1cm.
    """
    path = str(tmp_path / "utm.laz")
    known_x = np.array([500123.456, 500124.789])
    known_y = np.array([4000456.222, 4000457.333])
    known_z = np.array([1250.5, 1251.75])

    LAZWriter(path).write(_make_cloud(known_x, known_y, known_z))

    with laspy.open(path) as f:
        assert list(f.header.scales) == pytest.approx([0.001, 0.001, 0.001])

    with LAZReader(path) as reader:
        result = next(iter(reader.read()))

    np.testing.assert_allclose(result[PointAttribute.X], known_x, atol=1e-3)
    np.testing.assert_allclose(result[PointAttribute.Y], known_y, atol=1e-3)
    np.testing.assert_allclose(result[PointAttribute.Z], known_z, atol=1e-3)


def test_explicit_scale_and_offset_are_respected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "explicit.laz")
    cloud = _make_cloud(np.array([500000.1234]), np.array([4000000.5678]), np.array([100.0]))

    LAZWriter(path, scale=(0.0001, 0.0001, 0.0001), offset=(500000.0, 4000000.0, 0.0)).write(cloud)

    with laspy.open(path) as f:
        assert list(f.header.scales) == pytest.approx([0.0001, 0.0001, 0.0001])
        assert list(f.header.offsets) == pytest.approx([500000.0, 4000000.0, 0.0])


def test_file_is_genuinely_compressed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Confirms do_compress=True actually took effect (LAZ magic number)."""
    path = tmp_path / "compressed.laz"
    LAZWriter(str(path)).write(_make_cloud(np.array([1.0]), np.array([1.0]), np.array([1.0])))

    with laspy.open(str(path)) as f:
        assert f.header.are_points_compressed
