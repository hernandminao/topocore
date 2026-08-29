"""
Coverage audit tests for topocore.processing.features.geometric
(HeightFeatureComputer, DensityFeatureComputer,
DistanceToNeighborFeatureComputer). RelativeHeightFeatureComputer's
own batched-query behavior is already thoroughly covered by
test_relative_height_batched.py (PR21.8) -- only its own k<1
constructor validation and the reachable "missing CLASSIFICATION"
check (confirmed genuinely reachable, since CLASSIFICATION is an
optional attribute, unlike X/Y/Z which Chunk always guarantees) are
added here for completeness.

DistanceToNeighborFeatureComputer is confirmed orphaned -- zero
external callers via grep -- but exercised here as legitimate,
directly-testable public contract, consistent with this whole
session's established policy.

X/Y/Z "missing attribute" checks in both HeightFeatureComputer and
RelativeHeightFeatureComputer are NOT tested -- unreachable, matching
the same established pattern throughout this whole session
(Chunk.__init__ always requires X/Y/Z at construction).
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


def _cloud(n: int = 30, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# HeightFeatureComputer.
# ----------------------------------------------------------------------


def test_height_returns_z_coordinate_directly() -> None:
    cloud = _cloud()
    result = HeightFeatureComputer().compute(cloud)

    z = next(iter(cloud))[PointAttribute.Z]
    np.testing.assert_array_equal(result, z)


def test_height_metadata() -> None:
    computer = HeightFeatureComputer()
    assert computer.name() == "height"
    assert computer.dimension() == 1
    assert computer.requires_neighbors() is False
    assert computer.default_k() is None
    assert computer.default_radius() is None


# ----------------------------------------------------------------------
# RelativeHeightFeatureComputer -- constructor validation and the
# genuinely reachable "missing CLASSIFICATION" check.
# ----------------------------------------------------------------------


def test_relative_height_rejects_k_less_than_one() -> None:
    with pytest.raises(PointDescriptorError, match="at least 1"):
        RelativeHeightFeatureComputer(k=0)


def test_relative_height_rejects_missing_classification_attribute() -> None:
    """Unlike X/Y/Z (always guaranteed by Chunk), CLASSIFICATION is genuinely optional."""
    cloud = _cloud()
    with pytest.raises(PointDescriptorError, match="classification attribute"):
        RelativeHeightFeatureComputer().compute(cloud)


# ----------------------------------------------------------------------
# DensityFeatureComputer.
# ----------------------------------------------------------------------


def test_density_rejects_non_positive_radius() -> None:
    with pytest.raises(PointDescriptorError, match="must be positive"):
        DensityFeatureComputer(radius=0)


def test_density_happy_path_all_positive() -> None:
    cloud = _cloud()
    result = DensityFeatureComputer(radius=2.0).compute(cloud)

    assert result.shape == (30,)
    assert (result > 0).all()  # include_self=True guarantees at least the point itself


def test_density_metadata() -> None:
    computer = DensityFeatureComputer(radius=3.0)
    assert computer.name() == "density"
    assert computer.requires_neighbors() is True
    assert computer.default_radius() == 3.0
    assert computer.default_k() is None


# ----------------------------------------------------------------------
# DistanceToNeighborFeatureComputer.
# ----------------------------------------------------------------------


def test_distance_to_neighbor_rejects_k_less_than_one() -> None:
    with pytest.raises(PointDescriptorError, match="at least 1"):
        DistanceToNeighborFeatureComputer(k=0)


def test_distance_to_neighbor_increases_with_k() -> None:
    """The k-th nearest neighbor is always at least as far as the 1st."""
    cloud = _cloud()
    dist_k1 = DistanceToNeighborFeatureComputer(k=1).compute(cloud)
    dist_k3 = DistanceToNeighborFeatureComputer(k=3).compute(cloud)

    assert (dist_k1 <= dist_k3).all()


def test_distance_to_neighbor_metadata() -> None:
    computer = DistanceToNeighborFeatureComputer(k=2)
    assert computer.name() == "distance_to_neighbor_2"
    assert computer.dimension() == 1
    assert computer.requires_neighbors() is True
    assert computer.default_k() == 2
    assert computer.default_radius() is None
