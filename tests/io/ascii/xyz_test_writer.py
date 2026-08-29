"""
Coverage audit tests for topocore.io.ascii.xyz.writer.XYZWriter.

Performance investigation, NOT registered as a finding: write()'s
own per-point loop (`for xi, yi, zi in zip(...): stream.write(...)`)
superficially resembles the per-point Python loop pattern already
found and fixed elsewhere in this session (REG-ICP-001,
SAMPLING-DENSITY-001, SAMPLING-STRATIFIED-001). Benchmarked directly
before concluding anything: at n=200,000 points, the current
implementation (0.50s) is actually FASTER than a "vectorized"
`"\n".join(...)`-then-single-write alternative (0.67s), and
comparable to `np.savetxt` (0.55s). Python's own file buffering
already handles many small sequential writes efficiently here --
unlike the previous findings, no genuine performance problem exists,
so nothing is registered or changed.

zip(x, y, z, strict=True)'s own length-mismatch protection is
confirmed unreachable: Chunk guarantees all of its attributes share
the same length by construction (fixed `size` at construction time).

Confirmed direct round-trip compatibility with XYZReader.

XYZWriter is documented as orphaned -- zero real callers beyond its
own package's __init__.py re-export, confirmed via grep. Per this
audit's own established policy (an unused class is not automatically
a candidate for removal -- it may be prepared public API), it is
exercised here as legitimate, directly-testable public contract.
"""

from __future__ import annotations

from topocore.io.ascii.xyz.reader import XYZReader
from topocore.io.ascii.xyz.writer import XYZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud


def test_write_empty_cloud_produces_empty_file(tmp_path: object) -> None:
    path = str(tmp_path) + "/empty.xyz"  # type: ignore[operator]
    XYZWriter(path).write(PointCloud())

    with open(path) as f:
        content = f.read()
    assert content == ""


def test_write_multiple_chunks_preserves_order(tmp_path: object) -> None:
    cloud = PointCloud()
    chunk1 = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk1[PointAttribute.X][:] = [1.0, 2.0]
    chunk1[PointAttribute.Y][:] = [0.0, 0.0]
    chunk1[PointAttribute.Z][:] = [0.0, 0.0]
    chunk2 = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk2[PointAttribute.X][:] = [3.0, 4.0]
    chunk2[PointAttribute.Y][:] = [0.0, 0.0]
    chunk2[PointAttribute.Z][:] = [0.0, 0.0]
    cloud.add_chunk(chunk1)
    cloud.add_chunk(chunk2)

    path = str(tmp_path) + "/multi.xyz"  # type: ignore[operator]
    XYZWriter(path).write(cloud)

    with open(path) as f:
        lines = f.read().splitlines()
    assert lines == ["1.0 0.0 0.0", "2.0 0.0 0.0", "3.0 0.0 0.0", "4.0 0.0 0.0"]


def test_write_then_read_round_trip(tmp_path: object) -> None:
    cloud = PointCloud()
    chunk = Chunk(size=3, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.5, 2.5, 3.5]
    chunk[PointAttribute.Y][:] = [10.0, 20.0, 30.0]
    chunk[PointAttribute.Z][:] = [100.0, 200.0, 300.0]
    cloud.add_chunk(chunk)

    path = str(tmp_path) + "/round_trip.xyz"  # type: ignore[operator]
    XYZWriter(path).write(cloud)
    read_back = XYZReader(path).read()

    assert read_back.point_count == 3
    read_chunk = next(iter(read_back))
    import numpy as np

    np.testing.assert_allclose(read_chunk[PointAttribute.X], [1.5, 2.5, 3.5])
    np.testing.assert_allclose(read_chunk[PointAttribute.Y], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(read_chunk[PointAttribute.Z], [100.0, 200.0, 300.0])


def test_write_handles_negative_and_tiny_values(tmp_path: object) -> None:
    cloud = PointCloud()
    chunk = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [-123.456789012, 0.0000001]
    chunk[PointAttribute.Y][:] = [999999.999, -0.0]
    chunk[PointAttribute.Z][:] = [100.0, -50.5]
    cloud.add_chunk(chunk)

    path = str(tmp_path) + "/special.xyz"  # type: ignore[operator]
    XYZWriter(path).write(cloud)

    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0] == "-123.456789012 999999.999 100.0"
    assert lines[1] == "1e-07 -0.0 -50.5"


def test_close_does_not_raise(tmp_path: object) -> None:
    writer = XYZWriter(str(tmp_path) + "/x.xyz")  # type: ignore[operator]
    writer.close()  # ASCII writers keep no persistent resources open
