"""
Regression suite for topocore.io.laz.reader.LAZReader -- PR19.

Verified real chunked reading over an actual compressed .laz file
(lazrs backend): no points lost or duplicated across chunk
boundaries. No bugs found in this file (shares LASConverter, already
verified in tests/io/las/).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.io.exceptions import PointCloudIOError
from topocore.io.laz.reader import LAZReader
from topocore.io.laz.writer import LAZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud


def test_chunked_reading_no_lost_or_duplicated_points(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "chunks.laz")

    n = 250
    xs = np.arange(n, dtype=np.float64)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = xs * 2
    chunk[PointAttribute.Z][:] = xs * 3
    cloud.add_chunk(chunk)
    LAZWriter(path).write(cloud)

    with LAZReader(path, chunk_size=100) as reader:
        chunks = list(reader)
        all_x = np.concatenate([c[PointAttribute.X] for c in chunks])

    assert len(chunks) == 3
    assert len(all_x) == n
    np.testing.assert_allclose(np.sort(all_x), xs, atol=1e-3)


def test_read_returns_full_point_cloud(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "full.laz")
    cloud = PointCloud()
    chunk = Chunk(size=3, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 2.0, 3.0]
    chunk[PointAttribute.Y][:] = [1.0, 2.0, 3.0]
    chunk[PointAttribute.Z][:] = [1.0, 2.0, 3.0]
    cloud.add_chunk(chunk)
    LAZWriter(path).write(cloud)

    with LAZReader(path) as reader:
        result = reader.read()

    assert result.point_count == 3


def test_rejects_missing_file() -> None:
    with pytest.raises(PointCloudIOError), LAZReader("/nonexistent/path.laz") as reader:
        list(reader)


def test_rejects_nonpositive_chunk_size() -> None:
    with pytest.raises(ValueError):
        LAZReader("dummy.laz", chunk_size=0)
