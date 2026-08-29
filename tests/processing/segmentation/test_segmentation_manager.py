"""
Coverage audit tests for topocore.processing.segmentation.manager.SegmentationManager.

Phase 5.1-style (Integration) audit findings:

Unlike ClassificationManager (which has one real external caller,
workflow.py), SegmentationManager has ZERO real callers anywhere in
the codebase -- confirmed via grep. This is a whole-class orphan,
closer to ColumnMapper/math.numeric.py's status than
ClassificationManager's. Still tested here as legitimate, directly
constructible public API (no wrapper prevents it), and because this
phase's explicit purpose was to verify SEG-RG-001/SEG-SPEC-001
propagate correctly through this exact dispatcher.

set_params() and __call__ are additionally, individually orphaned
(zero callers) -- documented, not force-tested beyond a minimal
sanity check for set_params (confirmed reachable and functioning,
included as a straightforward public-contract test since it is
directly exercised by segment()'s own dispatch path).

_create_segmenter()'s final `raise RuntimeError(...)` is confirmed
unreachable: self._method can only ever be set via __init__ or the
method setter, both of which validate against _SUPPORTED_METHODS
first, and all 5 members of _SUPPORTED_METHODS are handled by an
explicit branch above the final raise.

Separate finding, NOT fixed here (a dispatch completeness gap, not a
coverage issue): DBSCANSegmenter's own `cache_neighbors` constructor
parameter is not forwarded by _create_segmenter()'s "dbscan" branch
-- confirmed directly that `SegmentationManager(method="dbscan",
cache_neighbors=False)` silently constructs a DBSCANSegmenter with
cache_neighbors=True (its class default) regardless. Flagged for a
separate decision, matching this audit's established discipline of
not fixing behavior gaps discovered incidentally during a coverage
pass.

Critical verification (the explicit purpose of this audit): confirmed
directly that both previously-fixed functional defects remain fixed
when reached through this manager's own public dispatch, not just
when the underlying classes are used directly:
  - SEG-RG-001 (region ID contiguity after filtering) via
    method="buildings" (which uses RegionGrowingSegmenter internally).
  - SEG-SPEC-001 (k<3 guard in cluster filtering) via method="trees".
These are captured below as permanent regression tests, not merely
ad-hoc verification.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.segmentation.manager import SegmentationManager

# ----------------------------------------------------------------------
# Constructor / method setter validation.
# ----------------------------------------------------------------------


def test_constructor_rejects_unsupported_method() -> None:
    with pytest.raises(SegmentationError, match="Unsupported method"):
        SegmentationManager(method="not_a_real_method")


def test_method_setter_rejects_unsupported_method() -> None:
    manager = SegmentationManager(method="dbscan")
    with pytest.raises(SegmentationError, match="Unsupported method"):
        manager.method = "not_a_real_method"


def test_method_setter_accepts_valid_method() -> None:
    manager = SegmentationManager(method="dbscan")
    manager.method = "region_growing"
    assert manager.method == "region_growing"


def test_set_params_applies_to_created_segmenter() -> None:
    manager = SegmentationManager(method="region_growing")
    manager.set_params(k=15)
    segmenter = manager._create_segmenter()
    assert segmenter._k == 15  # type: ignore[attr-defined]


# ----------------------------------------------------------------------
# segment() -- empty cloud.
# ----------------------------------------------------------------------


def test_segment_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError, match="empty point cloud"):
        SegmentationManager().segment(PointCloud())


# ----------------------------------------------------------------------
# Dispatch -- each of the 5 supported methods constructs the correct segmenter.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected_class_name"),
    [
        ("dbscan", "DBSCANSegmenter"),
        ("region_growing", "RegionGrowingSegmenter"),
        ("connected_components", "ConnectedComponentsSegmenter"),
        ("trees", "TreeSegmenter"),
        ("buildings", "BuildingSegmenter"),
    ],
)
def test_dispatch_creates_correct_segmenter_type(method: str, expected_class_name: str) -> None:
    manager = SegmentationManager(method=method)
    segmenter = manager._create_segmenter()
    assert type(segmenter).__name__ == expected_class_name


# ----------------------------------------------------------------------
# Critical regression: SEG-RG-001 and SEG-SPEC-001 fixes must remain
# effective when reached through this manager's own public dispatch.
# ----------------------------------------------------------------------


def test_seg_rg_001_fix_holds_through_buildings_dispatch() -> None:
    """BuildingSegmenter uses RegionGrowingSegmenter internally -- the contiguity fix must propagate through segment()."""
    rng = np.random.default_rng(0)
    gx, gy, gz = rng.uniform(0, 30, 3000), rng.uniform(0, 30, 3000), np.zeros(3000)
    roof_n = 80
    rx = np.full(roof_n, 15.0) + rng.normal(0, 0.5, roof_n)
    ry = np.full(roof_n, 15.0) + rng.normal(0, 0.5, roof_n)
    rz = np.full(roof_n, 5.0) + rng.normal(0, 0.02, roof_n)

    cloud = PointCloud()
    chunk = Chunk(size=3080, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = np.concatenate([gx, rx])
    chunk[PointAttribute.Y][:] = np.concatenate([gy, ry])
    chunk[PointAttribute.Z][:] = np.concatenate([gz, rz])
    cloud.add_chunk(chunk)

    manager = SegmentationManager(
        method="buildings",
        min_points_per_building=2,
        k=6,
        normal_angle_threshold=20.0,
        curvature_threshold=0.1,
    )

    result = manager.segment(cloud)  # must not raise a contiguity ValueError

    assert isinstance(result.num_segments, int)


def test_seg_spec_001_fix_holds_through_trees_dispatch() -> None:
    """The k<3 guard in _filter_tree_clusters() must remain effective when reached via segment()."""
    rng = np.random.default_rng(0)
    ground_n, tree_n = 200, 30
    gx, gy, gz = (
        rng.uniform(0, 50, ground_n),
        rng.uniform(0, 50, ground_n),
        np.zeros(ground_n),
    )
    tx = np.full(tree_n, 25.0) + rng.normal(0, 0.3, tree_n)
    ty = np.full(tree_n, 25.0) + rng.normal(0, 0.3, tree_n)
    tz = rng.uniform(1.0, 8.0, tree_n)

    cloud = PointCloud()
    chunk = Chunk(
        size=ground_n + tree_n,
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = np.concatenate([gx, tx])
    chunk[PointAttribute.Y][:] = np.concatenate([gy, ty])
    chunk[PointAttribute.Z][:] = np.concatenate([gz, tz])
    cloud.add_chunk(chunk)

    manager = SegmentationManager(method="trees", min_points_per_tree=1, eps=1.0, min_samples=3)

    result = manager.segment(cloud)  # must not raise PointDescriptorError

    assert isinstance(result.num_segments, int)
