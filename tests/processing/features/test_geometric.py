"""
Regression suite for topocore.processing.features.geometric -- PR19.

Includes a severe, real bug found and fixed in this session:
RelativeHeightFeatureComputer never read X/Y coordinates at all --
it built a "ground_points" array with ground Z values stuffed into
an X slot (zeros for Y/Z), then issued a SINGLE query for the whole
cloud using (0, 0, z[0]) instead of one query per point using each
point's own (x, y, z). Confirmed with a concrete reproduction: the
computed "relative height" was literally the raw Z value, unchanged,
for every point. Fixed to match the already-correct reference
pattern used in topocore.processing.ground.manager.GroundManager.
_nearest_ground_elevation() -- real per-point (x, y, z) ground
points, one query per point.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import PointDescriptorError
from topocore.processing.features.geometric import (
    DensityFeatureComputer,
    DistanceToNeighborFeatureComputer,
    HeightFeatureComputer,
    RelativeHeightFeatureComputer,
)


def _cloud(
    xs: list[float],
    ys: list[float],
    zs: list[float],
    cls: list[int] | None = None,
) -> PointCloud:
    attrs = [PointAttribute.X, PointAttribute.Y, PointAttribute.Z]
    if cls is not None:
        attrs.append(PointAttribute.CLASSIFICATION)

    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=attrs)
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    if cls is not None:
        chunk[PointAttribute.CLASSIFICATION][:] = cls
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# HeightFeatureComputer
# ----------------------------------------------------------------------


def test_height_returns_raw_z() -> None:
    cloud = _cloud([0.0, 1.0], [0.0, 0.0], [5.0, 10.0])
    result = HeightFeatureComputer().compute(cloud)
    np.testing.assert_array_equal(result, [5.0, 10.0])


def test_height_rejects_cloud_without_z() -> None:
    # Chunk itself always requires Z as a constructor invariant --
    # an empty (chunk-less) PointCloud is the only way to reach
    # HeightFeatureComputer's own "no Z" guard.
    empty_cloud = PointCloud()

    with pytest.raises(PointDescriptorError):
        HeightFeatureComputer().compute(empty_cloud)


# ----------------------------------------------------------------------
# RelativeHeightFeatureComputer -- the real bug.
# ----------------------------------------------------------------------


def test_relative_height_simple_case() -> None:
    cloud = _cloud(
        xs=[0.0, 1.0, 2.0, 0.5, 0.5],
        ys=[0.0, 0.0, 0.0, 0.0, 0.0],
        zs=[0.0, 1.0, 2.0, 5.0, 10.0],
        cls=[2, 2, 2, 6, 6],
    )

    result = RelativeHeightFeatureComputer(
        ground_class=2,
        k=1,
    ).compute(cloud)

    # Nearest ground point is determined using the complete XYZ
    # distance. For both non-ground points, (2, 0, 2) is nearest.
    np.testing.assert_allclose(
        result,
        [0.0, 0.0, 0.0, 3.0, 8.0],
    )

    # Regression guard: the old implementation returned raw Z.
    assert not np.array_equal(
        result,
        [0.0, 1.0, 2.0, 5.0, 10.0],
    )


def test_relative_height_uses_real_geometric_proximity() -> None:
    """
    Each point must query its own nearest ground point.

    Two ground clusters exist at different locations and elevations.
    The two non-ground points are deliberately placed close to
    different clusters.
    """
    cloud = _cloud(
        xs=[0.0, 0.0, 10.0, 10.0, 0.1, 10.1],
        ys=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        zs=[0.0, 0.0, 5.0, 5.0, 3.0, 3.0],
        cls=[2, 2, 2, 2, 6, 6],
    )

    result = RelativeHeightFeatureComputer(
        ground_class=2,
        k=1,
    ).compute(cloud)

    # (0.1, 0, 3) -> ground elevation 0
    assert result[4] == pytest.approx(3.0)

    # (10.1, 0, 3) -> ground elevation 5
    assert result[5] == pytest.approx(-2.0)


def test_relative_height_ground_points_are_zero_relative_to_themselves() -> None:
    cloud = _cloud(
        xs=[0.0, 5.0, 10.0],
        ys=[0.0, 0.0, 0.0],
        zs=[1.0, 2.0, 3.0],
        cls=[2, 2, 2],
    )

    result = RelativeHeightFeatureComputer(
        ground_class=2,
        k=1,
    ).compute(cloud)

    np.testing.assert_allclose(
        result,
        [0.0, 0.0, 0.0],
    )


def test_relative_height_averages_multiple_ground_neighbors() -> None:
    cloud = _cloud(
        xs=[0.0, 1.0, 0.5],
        ys=[0.0, 0.0, 0.0],
        zs=[0.0, 2.0, 10.0],
        cls=[2, 2, 6],
    )

    result = RelativeHeightFeatureComputer(
        ground_class=2,
        k=2,
    ).compute(cloud)

    expected_ground = 1.0
    expected_height = 10.0 - expected_ground

    assert result[2] == pytest.approx(expected_height)


def test_relative_height_rejects_invalid_k() -> None:
    with pytest.raises(PointDescriptorError):
        RelativeHeightFeatureComputer(k=0)


def test_relative_height_rejects_no_ground_points() -> None:
    cloud = _cloud([0.0], [0.0], [5.0], cls=[6])
    with pytest.raises(PointDescriptorError):
        RelativeHeightFeatureComputer(ground_class=2).compute(cloud)


def test_relative_height_rejects_missing_classification() -> None:
    cloud = _cloud([0.0], [0.0], [5.0])  # no cls attribute at all
    with pytest.raises(PointDescriptorError):
        RelativeHeightFeatureComputer(ground_class=2).compute(cloud)


# ----------------------------------------------------------------------
# DensityFeatureComputer
# ----------------------------------------------------------------------


def test_density_denser_cluster_has_higher_density() -> None:
    dense_xs = [float(i) * 0.1 for i in range(20)]
    dense_cloud = _cloud(dense_xs, [0.0] * 20, [0.0] * 20)

    sparse_xs = [float(i) * 2.0 for i in range(20)]
    sparse_cloud = _cloud(sparse_xs, [0.0] * 20, [0.0] * 20)

    dense_density = DensityFeatureComputer(radius=1.0).compute(dense_cloud)
    sparse_density = DensityFeatureComputer(radius=1.0).compute(sparse_cloud)

    assert dense_density.mean() > sparse_density.mean()


def test_density_rejects_nonpositive_radius() -> None:
    with pytest.raises(PointDescriptorError):
        DensityFeatureComputer(radius=0.0)


# ----------------------------------------------------------------------
# DistanceToNeighborFeatureComputer
# ----------------------------------------------------------------------


def test_distance_to_neighbor_known_spacing() -> None:
    # Points spaced exactly 1.0 apart on a line -- nearest neighbor
    # distance is exactly 1.0 for every interior point.
    xs = [float(i) for i in range(10)]
    cloud = _cloud(xs, [0.0] * 10, [0.0] * 10)

    result = DistanceToNeighborFeatureComputer(k=1).compute(cloud)
    np.testing.assert_allclose(result, [1.0] * 10)


def test_distance_to_neighbor_rejects_k_below_one() -> None:
    with pytest.raises(PointDescriptorError):
        DistanceToNeighborFeatureComputer(k=0)
