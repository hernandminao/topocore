"""
Regression suite for PR21.7.3: LAZWriter's rewritten internal
strategy -- mirrors topocore.io.las.writer.LASWriter's identical
fix. See that suite's own module docstring for the full rationale.

Per explicit instruction: LAZ compression is not required to be
byte-identical to any reference (two valid LAZ files encoding the
same points can legitimately differ in bytes) -- equivalence is
checked semantically, after reading both back.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import laspy  # type: ignore[import-untyped]
except ImportError:
    laspy = None

from topocore.io.laz.writer import LAZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud

pytestmark = pytest.mark.skipif(laspy is None, reason="laspy not installed")


def _multi_chunk_cloud_all_attributes(chunks: int = 3, points_per_chunk: int = 40, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    attrs = [
        PointAttribute.X,
        PointAttribute.Y,
        PointAttribute.Z,
        PointAttribute.INTENSITY,
        PointAttribute.CLASSIFICATION,
        PointAttribute.RETURN_NUMBER,
        PointAttribute.NUMBER_OF_RETURNS,
        PointAttribute.GPS_TIME,
        PointAttribute.COLOR,
    ]
    for _ in range(chunks):
        n = points_per_chunk
        chunk = Chunk(size=n, attributes=attrs)
        chunk[PointAttribute.X][:] = rng.uniform(500000, 500100, n)
        chunk[PointAttribute.Y][:] = rng.uniform(4000000, 4000100, n)
        chunk[PointAttribute.Z][:] = rng.uniform(100, 200, n)
        chunk[PointAttribute.INTENSITY][:] = rng.integers(0, 65535, n)
        chunk[PointAttribute.CLASSIFICATION][:] = rng.integers(0, 10, n)
        chunk[PointAttribute.RETURN_NUMBER][:] = rng.integers(1, 5, n)
        chunk[PointAttribute.NUMBER_OF_RETURNS][:] = rng.integers(1, 5, n)
        chunk[PointAttribute.GPS_TIME][:] = rng.uniform(0, 1000, n)
        chunk[PointAttribute.COLOR][:] = rng.integers(0, 65535, (n, 3))
        cloud.add_chunk(chunk)
    return cloud


def test_all_attributes_round_trip_correctly_semantic(tmp_path: object) -> None:
    cloud = _multi_chunk_cloud_all_attributes()
    path = str(tmp_path) + "/out.laz"  # type: ignore[operator]

    LAZWriter(path).write(cloud)
    result = laspy.read(path)

    expected_x = np.concatenate([c[PointAttribute.X] for c in cloud])
    expected_intensity = np.concatenate([c[PointAttribute.INTENSITY] for c in cloud])
    expected_color = np.concatenate([c[PointAttribute.COLOR] for c in cloud])

    assert np.allclose(result.x, expected_x)
    assert np.array_equal(result.intensity, expected_intensity)
    assert np.array_equal(result.red, expected_color[:, 0])
    assert np.array_equal(result.green, expected_color[:, 1])
    assert np.array_equal(result.blue, expected_color[:, 2])


def test_output_is_genuinely_compressed(tmp_path: object) -> None:
    cloud = _multi_chunk_cloud_all_attributes()
    path = str(tmp_path) + "/out.laz"  # type: ignore[operator]

    LAZWriter(path).write(cloud)
    result = laspy.read(path)

    assert result.header.are_points_compressed


def test_header_point_count_correct_after_multiple_chunks(tmp_path: object) -> None:
    cloud = _multi_chunk_cloud_all_attributes(chunks=5, points_per_chunk=17)
    path = str(tmp_path) + "/out.laz"  # type: ignore[operator]

    LAZWriter(path).write(cloud)
    result = laspy.read(path)

    assert result.header.point_count == 5 * 17
    assert len(result.points) == 5 * 17


def test_automatic_offset_matches_global_minimum(tmp_path: object) -> None:
    cloud = _multi_chunk_cloud_all_attributes()
    path = str(tmp_path) + "/out.laz"  # type: ignore[operator]

    LAZWriter(path).write(cloud)
    result = laspy.read(path)

    expected_min_x = min(float(np.min(c[PointAttribute.X])) for c in cloud)
    expected_min_y = min(float(np.min(c[PointAttribute.Y])) for c in cloud)
    expected_min_z = min(float(np.min(c[PointAttribute.Z])) for c in cloud)

    offsets = list(result.header.offsets)
    assert offsets[0] == pytest.approx(expected_min_x)
    assert offsets[1] == pytest.approx(expected_min_y)
    assert offsets[2] == pytest.approx(expected_min_z)


def test_automatic_offset_is_the_true_global_minimum_not_the_first_chunk(
    tmp_path: object,
) -> None:
    """
    The decisive, hand-constructed case: chunk 1's own minimum X
    (500) is LARGER than chunk 2's (100) and chunk 3's (300). If the
    per-chunk min/min/min reduction incorrectly only looked at the
    FIRST chunk (or otherwise depended on chunk order), this would
    wrongly produce offset X=500 instead of the true global minimum,
    100 -- confirming the reduction is order-independent, not just
    "first chunk wins".
    """
    cloud = PointCloud()
    for x_values in (
        [500.0, 510.0, 520.0],
        [100.0, 150.0, 200.0],
        [300.0, 350.0, 400.0],
    ):
        chunk = Chunk(size=3, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
        chunk[PointAttribute.X][:] = x_values
        chunk[PointAttribute.Y][:] = [1.0, 2.0, 3.0]
        chunk[PointAttribute.Z][:] = [10.0, 20.0, 30.0]
        cloud.add_chunk(chunk)

    path = str(tmp_path) + "/out.laz"  # type: ignore[operator]
    LAZWriter(path).write(cloud)
    result = laspy.read(path)

    offsets = list(result.header.offsets)
    assert offsets[0] == pytest.approx(100.0)  # the TRUE global min, from chunk 2, not chunk 1's 500.0


def test_explicit_offset_and_scale_still_honored(tmp_path: object) -> None:
    cloud = _multi_chunk_cloud_all_attributes()
    path = str(tmp_path) + "/out.laz"  # type: ignore[operator]

    LAZWriter(path, offset=(500000.0, 4000000.0, 100.0), scale=(0.01, 0.01, 0.01)).write(cloud)
    result = laspy.read(path)

    assert list(result.header.offsets) == pytest.approx([500000.0, 4000000.0, 100.0])
    assert list(result.header.scales) == pytest.approx([0.01, 0.01, 0.01])


def test_empty_cloud_produces_valid_zero_point_compressed_file(
    tmp_path: object,
) -> None:
    path = str(tmp_path) + "/out.laz"  # type: ignore[operator]

    LAZWriter(path).write(PointCloud())
    result = laspy.read(path)

    assert len(result.points) == 0


def test_single_chunk_matches_multi_chunk_with_same_data(tmp_path: object) -> None:
    rng = np.random.default_rng(42)
    n = 60
    xs = rng.uniform(500000, 500100, n)
    ys = rng.uniform(4000000, 4000100, n)
    zs = rng.uniform(100, 200, n)

    single = PointCloud()
    single_chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    single_chunk[PointAttribute.X][:] = xs
    single_chunk[PointAttribute.Y][:] = ys
    single_chunk[PointAttribute.Z][:] = zs
    single.add_chunk(single_chunk)

    multi = PointCloud()
    for start in (0, 20, 40):
        c = Chunk(size=20, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
        c[PointAttribute.X][:] = xs[start : start + 20]
        c[PointAttribute.Y][:] = ys[start : start + 20]
        c[PointAttribute.Z][:] = zs[start : start + 20]
        multi.add_chunk(c)

    path_single = str(tmp_path) + "/single.laz"  # type: ignore[operator]
    path_multi = str(tmp_path) + "/multi.laz"  # type: ignore[operator]
    LAZWriter(path_single).write(single)
    LAZWriter(path_multi).write(multi)

    result_single = laspy.read(path_single)
    result_multi = laspy.read(path_multi)

    np.testing.assert_allclose(result_single.x, result_multi.x)
    np.testing.assert_allclose(result_single.y, result_multi.y)
    np.testing.assert_allclose(result_single.z, result_multi.z)
