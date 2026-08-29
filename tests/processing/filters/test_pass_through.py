"""
Coverage audit tests for topocore.processing.filters.pass_through.PassThroughFilter.

Confirmed reachable, unlike several similar-looking checks audited
elsewhere in this whole session:
  - NaN/Inf in coordinate values: unlike compute_pca() (protected by
    scipy's own KDTree construction rejecting non-finite input
    before this kind of check could ever run), PassThroughFilter
    reads raw X/Y/Z attribute values directly with no such gate --
    confirmed directly that a NaN in the filtered axis is genuinely
    reachable and raises FilterError.
  - The ProcessingError-to-FilterError wrapping: confirmed reachable
    via heterogeneous chunks (add_chunk() does not validate
    attribute-set consistency, established earlier in this session's
    _shared.py audit) -- confirmed __cause__ is preserved.

Confirmed unreachable, matching established patterns:
  - "Point cloud has no {attribute} attribute": _AXIS_TO_ATTRIBUTE
    only maps to X/Y/Z, and Chunk.__init__ already requires all
    three at construction -- no PointCloud can ever lack them.
  - The post-flatten_attributes() length-mismatch check: given a
    successful (non-raising) flatten_attributes() call, its own
    already-audited contract guarantees every attribute has the same
    length by construction.

axis/min_value/max_value properties are documented as orphaned --
zero external callers confirmed via grep.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import FilterError, ProcessingError
from topocore.processing.filters.pass_through import Axis, PassThroughFilter


def _cloud(n: int = 30, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_min_greater_than_max_rejected() -> None:
    with pytest.raises(FilterError, match="must be <="):
        PassThroughFilter(Axis.X, min_value=10.0, max_value=5.0)


# ----------------------------------------------------------------------
# mask() -- empty cloud, happy path.
# ----------------------------------------------------------------------


def test_mask_rejects_empty_cloud() -> None:
    with pytest.raises(FilterError, match="empty point cloud"):
        PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0).mask(PointCloud())


def test_mask_happy_path_filters_by_axis_range() -> None:
    cloud = PointCloud()
    chunk = Chunk(size=5, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [0.0, 0.0, 0.0, 0.0, 0.0]
    chunk[PointAttribute.Y][:] = [0.0, 0.0, 0.0, 0.0, 0.0]
    chunk[PointAttribute.Z][:] = [-5.0, 0.0, 5.0, 10.0, 15.0]
    cloud.add_chunk(chunk)

    mask = PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0).mask(cloud)

    np.testing.assert_array_equal(mask, [False, True, True, True, False])


# ----------------------------------------------------------------------
# mask() -- NaN/Inf, confirmed genuinely reachable (no KDTree gate here).
# ----------------------------------------------------------------------


def test_mask_rejects_nan_in_filtered_axis() -> None:
    cloud = PointCloud()
    chunk = Chunk(size=5, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    chunk[PointAttribute.Y][:] = [0.0] * 5
    chunk[PointAttribute.Z][:] = [1.0, np.nan, 3.0, 4.0, 5.0]
    cloud.add_chunk(chunk)

    with pytest.raises(FilterError, match="NaN or Inf"):
        PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0).mask(cloud)


def test_mask_rejects_inf_in_filtered_axis() -> None:
    cloud = PointCloud()
    chunk = Chunk(size=3, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, np.inf, 3.0]
    chunk[PointAttribute.Y][:] = [0.0] * 3
    chunk[PointAttribute.Z][:] = [0.0] * 3
    cloud.add_chunk(chunk)

    with pytest.raises(FilterError, match="NaN or Inf"):
        PassThroughFilter(Axis.X, min_value=0.0, max_value=10.0).mask(cloud)


# ----------------------------------------------------------------------
# mask() -- ProcessingError wrapping, confirmed reachable via
# heterogeneous chunks.
# ----------------------------------------------------------------------


def test_mask_wraps_processing_error_from_heterogeneous_chunks() -> None:
    cloud = PointCloud()
    chunk_with_extra = Chunk(
        size=3,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.CLASSIFICATION,
        ],
    )
    chunk_with_extra[PointAttribute.X][:] = [1.0, 2.0, 3.0]
    chunk_with_extra[PointAttribute.Y][:] = [0.0] * 3
    chunk_with_extra[PointAttribute.Z][:] = [0.0] * 3
    chunk_with_extra[PointAttribute.CLASSIFICATION][:] = [0, 0, 0]

    chunk_plain = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk_plain[PointAttribute.X][:] = [4.0, 5.0]
    chunk_plain[PointAttribute.Y][:] = [0.0] * 2
    chunk_plain[PointAttribute.Z][:] = [0.0] * 2

    cloud.add_chunk(chunk_with_extra)
    cloud.add_chunk(chunk_plain)

    with pytest.raises(FilterError, match="Unable to flatten") as exc_info:
        PassThroughFilter(Axis.X, min_value=0.0, max_value=10.0).mask(cloud)

    assert isinstance(exc_info.value.__cause__, ProcessingError)
