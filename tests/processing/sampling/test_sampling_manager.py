"""
Coverage audit tests for topocore.processing.sampling.manager.SamplingManager.

SAMPLING-MANAGER-001 (FIXED in this PR): unlike the other 4 dispatch
branches (uniform, voxel, stratified, density), each of which
provides the underlying sampler's required parameter with a sensible
manager-level default (all 4 work with just
`SamplingManager(method=X).sample(cloud)`), "random" has no safe
universal default: a proportion or an absolute count both depend
entirely on the specific dataset, unlike a spatial unit (step,
voxel_size, cell_size, target_density), which is meaningful
independent of dataset size. Previously,
`SamplingManager(method="random").sample(cloud)` passed
`fraction=None, count=None` straight to RandomSampler, which fails
with "Specify exactly one of fraction or count, not both." -- a
confusing message given the caller specified NEITHER, not both.

Decision made explicitly (not invented unilaterally): no default
value is assumed for either parameter. Instead, SamplingManager now
raises its own clear SamplingError in the correct layer (the
manager, not RandomSampler) when both are omitted for "random".
RandomSampler's own validation is deliberately left untouched and
still fires as a second line of defense if this class is used
directly (not through the manager).

Confirmed directly, before and after this change: the other 4
methods' own "works out of the box" behavior is completely
unaffected, and explicitly passing `fraction=` or `count=` through
the manager still produces the exact same results as before.

Confirmed unreachable, matching the same pattern already established
for segmentation/manager.py's own analogous check: the final
`raise SamplingError(f"Unsupported method: {self._method}")` in
_create_sampler() -- self._method can only ever be set via __init__
or the method setter, both of which validate against
_SUPPORTED_METHODS first, and all 5 members are handled by an
explicit branch above the final raise.

Confirmed no parameter-propagation gaps exist for the other 4
methods -- each dispatched sampler class's own full constructor
signature was checked directly against what _create_sampler()
passes, with no gap found (unlike SEG-MANAGER-001's cache_neighbors
omission).

__call__ and SamplingManager as a whole class are documented as
orphaned -- zero external callers confirmed via grep.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.density import DensitySampler
from topocore.processing.sampling.manager import SamplingManager
from topocore.processing.sampling.stratified import StratifiedSampler
from topocore.processing.sampling.uniform import UniformSampler
from topocore.processing.sampling.voxel import VoxelSampler


def _cloud(n: int = 200, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Constructor / method setter validation.
# ----------------------------------------------------------------------


def test_constructor_rejects_unsupported_method() -> None:
    with pytest.raises(SamplingError, match="Unsupported method"):
        SamplingManager(method="not_a_real_method")


def test_method_setter_rejects_unsupported_method() -> None:
    manager = SamplingManager(method="voxel")
    with pytest.raises(SamplingError, match="Unsupported method"):
        manager.method = "not_a_real_method"


def test_method_setter_accepts_valid_method() -> None:
    manager = SamplingManager(method="voxel")
    manager.method = "uniform"
    assert manager.method == "uniform"


def test_set_params_applies_to_created_sampler() -> None:
    manager = SamplingManager(method="uniform")
    manager.set_params(step=5)
    sampler = manager._create_sampler()
    assert isinstance(sampler, UniformSampler)
    assert sampler._step == 5


# ----------------------------------------------------------------------
# Dispatch -- the 4 methods confirmed to work out of the box with
# sensible manager-level defaults.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected_class"),
    [
        ("uniform", UniformSampler),
        ("voxel", VoxelSampler),
        ("stratified", StratifiedSampler),
        ("density", DensitySampler),
    ],
)
def test_dispatch_with_default_params_works_out_of_the_box(method: str, expected_class: type) -> None:
    cloud = _cloud()
    result = SamplingManager(method=method).sample(cloud)

    assert 0 < result.point_count <= cloud.point_count
    assert isinstance(SamplingManager(method=method)._create_sampler(), expected_class)


def test_random_without_fraction_or_count_raises_clear_manager_error() -> None:
    """
    The core regression: SamplingManager itself now raises a clear,
    manager-level SamplingError -- not RandomSampler's own confusing
    "not both" message -- when neither fraction nor count is given.
    """
    cloud = _cloud()
    with pytest.raises(SamplingError, match="requires an explicit 'fraction' or 'count'"):
        SamplingManager(method="random").sample(cloud)


def test_random_dispatch_with_explicit_fraction_still_works() -> None:
    """Explicitly providing fraction/count through the manager is unaffected by the new validation."""
    cloud = _cloud()
    result = SamplingManager(method="random").sample(cloud, fraction=0.3, seed=42)
    assert result.point_count == 60


def test_random_dispatch_with_explicit_count_still_works() -> None:
    cloud = _cloud()
    result = SamplingManager(method="random").sample(cloud, count=25, seed=42)
    assert result.point_count == 25
