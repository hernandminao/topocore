"""
Coverage audit tests for topocore.processing.segmentation.base.SegmentationResult
and Segmenter.

SegmentationResult is a public, general-purpose dataclass with no
wrapper preventing malformed construction -- confirmed 5 real
construction sites (region_growing.py, dbscan.py,
connected_components.py, specific.py x2), matching the same
reasoning as processing._shared.build_cloud()'s own validation (see
that audit): every branch is genuinely testable regardless of what
today's callers happen to construct.

get_segment() is confirmed genuinely active: 2 real call sites in
segmentation/specific.py.

NOT tested here by design -- documented as orphaned:
get_segments() (appears ONLY in docstring examples across three
files, never in actually-executed code -- confirmed via grep),
has_noise (zero references anywhere, not even in docstrings), and
Segmenter.__call__ (zero usage of the callable-interface pattern,
same orphaned pattern as NormalManager.__call__/Classifier.__call__
found elsewhere in this audit).

The abstract classes (Segmenter, ClusterSegmenter,
RegionGrowingSegmenter) are never directly instantiated; their
abstract method declarations are exercised through concrete
subclasses (DBSCANSegmenter, etc.), which have their own coverage
elsewhere -- nothing to test here for the interface declarations
themselves.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.segmentation.base import SegmentationResult


def _cloud(n: int) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = np.arange(n, dtype=float)
    chunk[PointAttribute.Y][:] = np.zeros(n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Happy path -- including the num_segments == 0 (all-noise) edge case,
# a genuine, legitimate outcome (e.g. DBSCAN classifying everything as
# noise), not merely a degenerate input.
# ----------------------------------------------------------------------


def test_happy_path_with_two_segments() -> None:
    result = SegmentationResult(
        labels=np.array([0, 0, 1, 1, 1]),
        num_segments=2,
        segment_sizes=np.array([2, 3]),
        cloud=_cloud(5),
    )
    assert result.has_noise is False


def test_all_noise_zero_segments_is_valid() -> None:
    result = SegmentationResult(
        labels=np.array([-1, -1, -1, -1, -1]),
        num_segments=0,
        segment_sizes=np.array([], dtype=np.int64),
        cloud=_cloud(5),
    )
    assert result.has_noise is True


# ----------------------------------------------------------------------
# get_segment() -- confirmed active via 2 real call sites.
# ----------------------------------------------------------------------


def test_get_segment_extracts_correct_points() -> None:
    result = SegmentationResult(
        labels=np.array([0, 0, 1, 1, 1]),
        num_segments=2,
        segment_sizes=np.array([2, 3]),
        cloud=_cloud(5),
    )

    segment = result.get_segment(0)

    assert segment.point_count == 2


def test_get_segment_rejects_out_of_range_id() -> None:
    result = SegmentationResult(
        labels=np.array([0, 0, 1, 1, 1]),
        num_segments=2,
        segment_sizes=np.array([2, 3]),
        cloud=_cloud(5),
    )

    with pytest.raises(ValueError, match="must be in"):
        result.get_segment(5)


def test_get_segment_rejects_non_int_id() -> None:
    result = SegmentationResult(
        labels=np.array([0, 0, 1, 1, 1]),
        num_segments=2,
        segment_sizes=np.array([2, 3]),
        cloud=_cloud(5),
    )

    with pytest.raises(TypeError, match="must be an int"):
        result.get_segment("0")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# __post_init__ validation -- num_segments.
# ----------------------------------------------------------------------


def test_num_segments_non_int_rejected() -> None:
    with pytest.raises(TypeError, match="must be an int"):
        SegmentationResult(
            labels=np.array([0]),
            num_segments="2",  # type: ignore[arg-type]
            segment_sizes=np.array([1]),
            cloud=_cloud(1),
        )


def test_num_segments_negative_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SegmentationResult(
            labels=np.array([0]),
            num_segments=-1,
            segment_sizes=np.array([1]),
            cloud=_cloud(1),
        )


# ----------------------------------------------------------------------
# __post_init__ validation -- labels.
# ----------------------------------------------------------------------


def test_labels_two_dimensional_rejected() -> None:
    with pytest.raises(ValueError, match="1D array"):
        SegmentationResult(
            labels=np.array([[0]]),
            num_segments=1,
            segment_sizes=np.array([1]),
            cloud=_cloud(1),
        )


def test_labels_non_integer_rejected() -> None:
    with pytest.raises(TypeError, match="integer dtype"):
        SegmentationResult(
            labels=np.array([0.0]),
            num_segments=1,
            segment_sizes=np.array([1]),
            cloud=_cloud(1),
        )


def test_labels_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="length must match"):
        SegmentationResult(
            labels=np.array([0, 0]),
            num_segments=1,
            segment_sizes=np.array([2]),
            cloud=_cloud(1),
        )


# ----------------------------------------------------------------------
# __post_init__ validation -- segment_sizes.
# ----------------------------------------------------------------------


def test_segment_sizes_two_dimensional_rejected() -> None:
    with pytest.raises(ValueError, match="1D array"):
        SegmentationResult(
            labels=np.array([0]),
            num_segments=1,
            segment_sizes=np.array([[1]]),
            cloud=_cloud(1),
        )


def test_segment_sizes_non_integer_rejected() -> None:
    with pytest.raises(TypeError, match="integer dtype"):
        SegmentationResult(
            labels=np.array([0]),
            num_segments=1,
            segment_sizes=np.array([1.0]),
            cloud=_cloud(1),
        )


def test_segment_sizes_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        SegmentationResult(
            labels=np.array([0, 1]),
            num_segments=1,
            segment_sizes=np.array([1, 1]),
            cloud=_cloud(2),
        )


def test_segment_sizes_negative_rejected() -> None:
    with pytest.raises(ValueError, match="cannot contain negative"):
        SegmentationResult(
            labels=np.array([0]),
            num_segments=1,
            segment_sizes=np.array([-1]),
            cloud=_cloud(1),
        )


# ----------------------------------------------------------------------
# __post_init__ validation -- label domain.
# ----------------------------------------------------------------------


def test_zero_segments_with_non_noise_label_rejected() -> None:
    with pytest.raises(ValueError, match="all labels must be -1"):
        SegmentationResult(
            labels=np.array([0]),
            num_segments=0,
            segment_sizes=np.array([], dtype=np.int64),
            cloud=_cloud(1),
        )


def test_label_outside_valid_range_rejected() -> None:
    with pytest.raises(ValueError, match="outside the valid range"):
        SegmentationResult(
            labels=np.array([5]),
            num_segments=2,
            segment_sizes=np.array([1, 0]),
            cloud=_cloud(1),
        )


# ----------------------------------------------------------------------
# __post_init__ validation -- consistency and contiguity.
# ----------------------------------------------------------------------


def test_segment_sizes_inconsistent_with_labels_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent with labels"):
        SegmentationResult(
            labels=np.array([0, 0, 1]),
            num_segments=2,
            segment_sizes=np.array([1, 1]),
            cloud=_cloud(3),
        )


def test_non_contiguous_segment_ids_rejected() -> None:
    with pytest.raises(ValueError, match="must be contiguous"):
        SegmentationResult(
            labels=np.array([0, 0, 2]),
            num_segments=3,
            segment_sizes=np.array([2, 0, 1]),
            cloud=_cloud(3),
        )
