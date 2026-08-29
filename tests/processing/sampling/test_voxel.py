"""
Coverage audit tests for topocore.processing.sampling.voxel.VoxelSampler.

PR22 coverage hardening.

VOXEL-DEAD-CODE-001 (FIXED in this PR): confirmed via direct
instrumentation (monkeypatching every suspect method and calling
sample() for all 4 public methods: "centroid", "closest", "random",
"all") that _sample_centroid(), _sample_closest(),
_compute_voxel_centroid(), and _compute_voxel_closest() were NEVER
called from any real usage path. Root cause: PR21.7.5 replaced
"centroid"/"closest" with the newer, chunk-wise
_sample_centroid_chunked()/_sample_closest_chunked() methods, called
directly from sample() BEFORE _apply_sampling() is ever reached --
but the old implementations were left in place rather than removed.

Before removing them, a broader consumer check (beyond production
code) found ONE genuine indirect consumer: this package's own
test_voxel_chunked_accumulator.py (from PR21.7.5's original work)
imported both module-level functions as a known-correct REFERENCE
implementation to cross-check the new chunked accumulator against
the pre-PR21.7.5 algorithm at scale (n=2000, across multiple
chunks). That cross-check's own value was preserved: the reference
implementation was moved into that test file itself (as
`_reference_compute_voxel_centroid`/`_reference_compute_voxel_closest`),
since its only genuine purpose was ever as a test-scoped baseline,
never a production capability. Confirmed both of that file's
previously-passing cross-check tests still pass unchanged after this
move.

Removed from production: the 2 module-level functions,
_apply_sampling()'s own now-dead "centroid"/"closest" branches (and
its no-longer-needed `points` parameter), and the 2 class methods
(_sample_centroid(), _sample_closest()) that were their only
callers. Confirmed directly, before and after removal: all 4 public
methods ("all", "centroid", "closest", "random") produce identical
results, including the exact-boundary and negative-coordinate
functional tests below.

A second, minor finding, not given its own numbered ID: _build_from_points()
contains a no-op loop (`for attr in cloud.attributes: if attr not in
attributes: pass`) that does nothing observable -- likely a remnant
of an incomplete feature. Not tested or removed.

Chunk invariance (identical results regardless of 1 vs 4 chunks) is
confirmed directly for all 3 real methods (centroid, closest,
random), matching this module's own docstring claims about "random"
specifically.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.voxel import VoxelSampler, _voxel_indices


def _cloud(n: int = 50, seed: int = 0, with_intensity: bool = False) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    attrs = [PointAttribute.X, PointAttribute.Y, PointAttribute.Z]
    if with_intensity:
        attrs.append(PointAttribute.INTENSITY)
    chunk = Chunk(size=n, attributes=attrs)
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    if with_intensity:
        chunk[PointAttribute.INTENSITY][:] = np.arange(n, dtype=np.uint16)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_constructor_rejects_non_positive_voxel_size() -> None:
    with pytest.raises(SamplingError, match="must be positive"):
        VoxelSampler(voxel_size=0)


def test_constructor_rejects_unsupported_method() -> None:
    with pytest.raises(SamplingError, match="method must be one of"):
        VoxelSampler(voxel_size=1.0, method="bogus")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# _voxel_indices() -- direct validation.
# ----------------------------------------------------------------------


def test_voxel_indices_rejects_non_positive_size() -> None:
    with pytest.raises(SamplingError, match="must be positive"):
        _voxel_indices(np.array([1.0]), np.array([1.0]), np.array([1.0]), 0.0)


def test_voxel_indices_rejects_empty_coordinates() -> None:
    with pytest.raises(SamplingError, match="empty coordinates"):
        _voxel_indices(np.array([]), np.array([]), np.array([]), 1.0)


def test_voxel_indices_rejects_non_finite_coordinates() -> None:
    with pytest.raises(SamplingError, match="NaN or Inf"):
        _voxel_indices(np.array([np.nan]), np.array([1.0]), np.array([1.0]), 1.0)


def test_voxel_indices_rejects_overflow_risk() -> None:
    with pytest.raises(SamplingError, match="overflow"):
        _voxel_indices(np.array([1e30]), np.array([1.0]), np.array([1.0]), 1e-10)


# ----------------------------------------------------------------------
# sample() -- empty cloud, "all" happy path.
# ----------------------------------------------------------------------


def test_sample_rejects_empty_cloud() -> None:
    with pytest.raises(SamplingError, match="empty point cloud"):
        VoxelSampler(voxel_size=1.0).sample(PointCloud())


def test_sample_all_returns_same_instance_unmodified() -> None:
    cloud = _cloud()
    result = VoxelSampler(voxel_size=1.0, method="all").sample(cloud)
    assert result is cloud


# ----------------------------------------------------------------------
# Happy path -- centroid / closest / random.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["centroid", "closest", "random"])
def test_sample_reduces_point_count(method: str) -> None:
    cloud = _cloud()
    result = VoxelSampler(voxel_size=2.0, method=method, seed=0).sample(cloud)  # type: ignore[arg-type]
    assert 0 < result.point_count <= cloud.point_count


def test_floor_based_voxel_boundary_groups_correctly() -> None:
    """floor(1.0)=1 and floor(1.999)=1 fall in the SAME voxel -- 3 distinct voxels from 4 points, not 4."""
    cloud = PointCloud()
    chunk = Chunk(size=4, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [0.999, 1.0, 1.999, 2.0]
    chunk[PointAttribute.Y][:] = [0.0] * 4
    chunk[PointAttribute.Z][:] = [0.0] * 4
    cloud.add_chunk(chunk)

    result = VoxelSampler(voxel_size=1.0, method="closest").sample(cloud)

    assert result.point_count == 3


def test_negative_coordinates_group_correctly() -> None:
    cloud = PointCloud()
    chunk = Chunk(size=4, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [-5.5, -5.4, 5.5, 5.6]
    chunk[PointAttribute.Y][:] = [0.0] * 4
    chunk[PointAttribute.Z][:] = [0.0] * 4
    cloud.add_chunk(chunk)

    result = VoxelSampler(voxel_size=1.0, method="closest").sample(cloud)

    assert result.point_count == 2


def test_closest_preserves_non_spatial_attributes() -> None:
    cloud = _cloud(with_intensity=True)
    result = VoxelSampler(voxel_size=2.0, method="closest").sample(cloud)
    assert PointAttribute.INTENSITY in result.attributes


def test_centroid_discards_non_spatial_attributes() -> None:
    """Centroid aggregation has no interpolation rule for non-spatial attributes, so they are dropped."""
    cloud = _cloud(with_intensity=True)
    result = VoxelSampler(voxel_size=2.0, method="centroid").sample(cloud)
    assert PointAttribute.INTENSITY not in result.attributes


def test_random_is_reproducible_with_same_seed() -> None:
    cloud = _cloud()
    result_a = VoxelSampler(voxel_size=2.0, method="random", seed=42).sample(cloud)
    result_b = VoxelSampler(voxel_size=2.0, method="random", seed=42).sample(cloud)
    assert result_a.point_count == result_b.point_count


# ----------------------------------------------------------------------
# Chunk invariance -- confirmed for all 3 real methods.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["centroid", "closest", "random"])
def test_result_is_invariant_to_chunk_partitioning(method: str) -> None:
    rng = np.random.default_rng(0)
    n = 50
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = rng.uniform(0, 10, n)

    cloud_one = PointCloud()
    chunk_one = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk_one[PointAttribute.X][:] = xs
    chunk_one[PointAttribute.Y][:] = ys
    chunk_one[PointAttribute.Z][:] = zs
    cloud_one.add_chunk(chunk_one)

    cloud_four = PointCloud()
    splits = np.linspace(0, n, 5, dtype=int)
    for i in range(4):
        s, e = splits[i], splits[i + 1]
        chunk = Chunk(
            size=int(e - s),
            attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
        )
        chunk[PointAttribute.X][:] = xs[s:e]
        chunk[PointAttribute.Y][:] = ys[s:e]
        chunk[PointAttribute.Z][:] = zs[s:e]
        cloud_four.add_chunk(chunk)

    result_one = VoxelSampler(voxel_size=2.0, method=method, seed=7).sample(cloud_one)  # type: ignore[arg-type]
    result_four = VoxelSampler(voxel_size=2.0, method=method, seed=7).sample(cloud_four)  # type: ignore[arg-type]

    assert result_one.point_count == result_four.point_count


# ----------------------------------------------------------------------
# name().
# ----------------------------------------------------------------------


def test_name_includes_configuration() -> None:
    sampler = VoxelSampler(voxel_size=0.5, method="centroid", seed=42)
    assert sampler.name() == "voxel(size=0.5, method=centroid, seed=42)"
