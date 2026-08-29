"""
Coverage audit tests for topocore.processing.sampling.uniform.UniformSampler.

Uses a single vectorized np.arange() call, not a per-point Python
loop -- unlike DensitySampler/StratifiedSampler's own audits, there
is no performance finding here.

Constructor validations (step < 1, offset out of range) were already
covered by the existing test suite (not in the original coverage
report's missing-lines list) -- only the empty-cloud check and
name() were missing.

name() is documented as orphaned -- zero external callers confirmed
via grep.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.uniform import UniformSampler


def _cloud(n: int = 10) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = np.arange(n, dtype=float)
    chunk[PointAttribute.Y][:] = np.zeros(n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)
    return cloud


def test_sample_rejects_empty_cloud() -> None:
    with pytest.raises(SamplingError, match="empty point cloud"):
        UniformSampler(step=10).sample(PointCloud())


def test_sample_with_step_and_offset() -> None:
    result = UniformSampler(step=3, offset=1).sample(_cloud())

    (chunk,) = list(result)
    np.testing.assert_array_equal(chunk[PointAttribute.X], [1.0, 4.0, 7.0])


def test_step_one_keeps_all_points() -> None:
    result = UniformSampler(step=1).sample(_cloud())
    assert result.point_count == 10
