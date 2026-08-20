"""
Regression suite for topocore.io.las.reader.LASReader -- PR19.

Verified real chunked reading over an actual multi-chunk LAS file:
no points lost or duplicated across chunk boundaries. No bugs found.
"""

from __future__ import annotations

import laspy  # type: ignore[import-untyped]
import numpy as np
import pytest

from topocore.io.exceptions import PointCloudIOError
from topocore.io.las.reader import LASReader
from topocore.pointcloud.attributes import PointAttribute


def test_chunked_reading_no_lost_or_duplicated_points(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "chunks.las")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)

    n = 250
    xs = np.arange(n, dtype=np.float64)
    las.x = xs
    las.y = xs * 2
    las.z = xs * 3
    las.write(path)

    with LASReader(path, chunk_size=100) as reader:
        chunks = list(reader)
        all_x = np.concatenate([c[PointAttribute.X] for c in chunks])

    assert len(chunks) == 3  # 100, 100, 50
    assert len(all_x) == n
    np.testing.assert_allclose(np.sort(all_x), xs)


def test_read_returns_full_point_cloud(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "full.las")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    las.x = np.array([1.0, 2.0, 3.0])
    las.y = np.array([1.0, 2.0, 3.0])
    las.z = np.array([1.0, 2.0, 3.0])
    las.write(path)

    with LASReader(path) as reader:
        cloud = reader.read()

    assert cloud.point_count == 3


def test_rejects_missing_file() -> None:
    with pytest.raises(PointCloudIOError), LASReader("/nonexistent/path.las") as reader:
        list(reader)


def test_rejects_nonpositive_chunk_size() -> None:
    with pytest.raises(ValueError):
        LASReader("dummy.las", chunk_size=0)


def test_context_manager_closes_reader(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "close.las")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    las.x = np.array([1.0])
    las.y = np.array([1.0])
    las.z = np.array([1.0])
    las.write(path)

    reader = LASReader(path)
    with reader:
        list(reader)

    assert reader._reader is None
