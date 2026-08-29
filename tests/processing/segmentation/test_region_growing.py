"""
Coverage audit tests for topocore.processing.segmentation.region_growing.RegionGrowingSegmenter.

NEW FINDING during this audit, registered as SEG-RG-001 -- NOT fixed,
NOT frozen by a test: _filter_small_regions() marks regions smaller
than min_region_size as noise (-1) WITHOUT renumbering the surviving
regions to remain contiguous from 0. Confirmed reproducible directly:
when the first-discovered region (lowest curvature, assigned id 0)
is filtered out as too small but a later region (id 1) survives, the
final labels contain {-1, 1} -- not contiguous -- and
SegmentationResult's own __post_init__ validation correctly rejects
this with ValueError("Segment IDs must be contiguous..."). This is a
genuine functional defect (not merely architectural debt), matching
the same severity class as SEG-SPEC-001 found earlier in this audit.
The happy-path tests below are deliberately designed (via
min_region_size=1, so nothing gets filtered) to avoid triggering it.

Multiple branches confirmed unreachable, chained through
_compute_normals() running BEFORE _compute_growth_radius() in
segment()'s own execution order: _compute_normals() already requires
cloud.point_count >= k (via NormalManager/compute_pca(), k >= 3 by
this class's own constructor validation), which means by the time
_compute_growth_radius() runs, n_points is guaranteed >= 3 --
making its own `n_points <= 1` check unreachable, and consequently
`k < 1` (computed as min(self._k, n_points-1) >= min(3, 2) = 2)
mathematically unreachable too. The `mean_distance <= 0.0` check was
not proven unreachable with the same certainty, but the one
coincident-points scenario tried failed earlier inside
_compute_normals() itself (a NeighborError from the KDTree), before
ever reaching this check -- not tested here.

_compute_normals()'s "has NORMAL attribute but no normal data
chunks" branch is unreachable for the same reason established
elsewhere in this audit for point_to_plane.py's own _ensure_normals():
an empty cloud's .attributes is an empty frozenset, so this branch is
only reached when at least one chunk already declares NORMAL,
guaranteeing normals_chunks is non-empty.

seed_points(), smoothness_criterion(), and the `name` property are
documented as orphaned -- zero external callers confirmed via grep.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.segmentation.region_growing import RegionGrowingSegmenter


def _flat_plane(n: int = 30, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_k_less_than_three_rejected() -> None:
    with pytest.raises(SegmentationError, match="k must be"):
        RegionGrowingSegmenter(k=2)


def test_negative_curvature_threshold_rejected() -> None:
    with pytest.raises(SegmentationError, match="curvature_threshold must be"):
        RegionGrowingSegmenter(curvature_threshold=-1.0)


def test_normal_angle_threshold_out_of_range_rejected() -> None:
    with pytest.raises(SegmentationError, match="normal_angle_threshold must be"):
        RegionGrowingSegmenter(normal_angle_threshold=100.0)


def test_min_region_size_less_than_one_rejected() -> None:
    with pytest.raises(SegmentationError, match="min_region_size must be"):
        RegionGrowingSegmenter(min_region_size=0)


def test_max_region_size_less_than_min_rejected() -> None:
    with pytest.raises(SegmentationError, match="must be >= min_region_size"):
        RegionGrowingSegmenter(min_region_size=10, max_region_size=5)


# ----------------------------------------------------------------------
# segment() -- empty cloud.
# ----------------------------------------------------------------------


def test_segment_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError, match="empty point cloud"):
        RegionGrowingSegmenter().segment(PointCloud())


# ----------------------------------------------------------------------
# _require_*() -- called before compute.
# ----------------------------------------------------------------------


def test_require_normals_before_computed_raises() -> None:
    with pytest.raises(SegmentationError, match="Normals have not been computed"):
        RegionGrowingSegmenter()._require_normals()


def test_require_curvature_before_computed_raises() -> None:
    with pytest.raises(SegmentationError, match="Curvature has not been computed"):
        RegionGrowingSegmenter()._require_curvature()


def test_require_labels_before_computed_raises() -> None:
    with pytest.raises(SegmentationError, match="labels have not been initialized"):
        RegionGrowingSegmenter()._require_labels()


# ----------------------------------------------------------------------
# _compute_normals() -- both branches.
# ----------------------------------------------------------------------


def test_compute_normals_uses_existing_normal_attribute() -> None:
    cloud = PointCloud()
    chunk = Chunk(
        size=10,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.NORMAL,
        ],
    )
    chunk[PointAttribute.X][:] = np.arange(10, dtype=float)
    chunk[PointAttribute.Y][:] = np.zeros(10)
    chunk[PointAttribute.Z][:] = np.zeros(10)
    chunk[PointAttribute.NORMAL][:] = np.tile([0.0, 0.0, 1.0], (10, 1))
    cloud.add_chunk(chunk)

    seg = RegionGrowingSegmenter(k=5, min_region_size=1)
    seg._compute_normals(cloud)

    np.testing.assert_array_equal(seg._normals, np.tile([0.0, 0.0, 1.0], (10, 1)))


def test_compute_normals_computes_when_missing() -> None:
    cloud = _flat_plane(30)
    seg = RegionGrowingSegmenter(k=5, min_region_size=1)

    seg._compute_normals(cloud)

    assert seg._normals is not None
    assert seg._normals.shape == (30, 3)


# ----------------------------------------------------------------------
# Happy path -- deliberately using min_region_size=1 to avoid triggering
# SEG-RG-001 (no region gets filtered, so no contiguity gap can occur).
# ----------------------------------------------------------------------


def test_segment_flat_plane_forms_one_region() -> None:
    cloud = _flat_plane(30)

    result = RegionGrowingSegmenter(
        k=5,
        min_region_size=1,
        curvature_threshold=0.5,
        normal_angle_threshold=45.0,
    ).segment(cloud)

    assert result.num_segments == 1
    assert result.has_noise is False


def test_max_region_size_caps_region_growth() -> None:
    cloud = _flat_plane(100)

    result = RegionGrowingSegmenter(
        k=5,
        min_region_size=1,
        max_region_size=20,
        curvature_threshold=0.5,
        normal_angle_threshold=45.0,
    ).segment(cloud)

    assert all(size <= 20 for size in result.segment_sizes)
    assert max(result.segment_sizes) == 20  # confirms the cap is genuinely reached, not just respected


def test_min_region_size_too_high_marks_everything_as_noise() -> None:
    """A min_region_size no real region can reach -- confirmed reachable, and safe from SEG-RG-001 (nothing survives to create a gap)."""
    cloud = _flat_plane(30)

    result = RegionGrowingSegmenter(k=5, min_region_size=1000).segment(cloud)

    assert result.num_segments == 0
    assert result.has_noise is True
