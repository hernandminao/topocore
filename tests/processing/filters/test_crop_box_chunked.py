"""
Regression suite for PR21.7.4: CropBoxFilter.mask()'s rewritten
per-chunk strategy -- computing the boolean mask on each Chunk's own
X/Y/Z arrays directly, then concatenating only the resulting
(smaller) per-chunk boolean masks, instead of first concatenating
every chunk's X/Y/Z into three giant float64 arrays.

The public contract (CropBoxFilter(box).mask(cloud)) is completely
unchanged; only the internal implementation strategy changed. This
suite verifies numerical equivalence to what the pre-PR21.7.4
concatenate-everything approach produced, plus the specific concern
this PR's own design discussion raised explicitly: a point selected
in a LATER chunk (not the first) must be correctly reflected at its
correct GLOBAL position in the returned mask -- confirming the
per-chunk masks are concatenated in the right order, not just
"first chunk wins" or otherwise order-dependent.

See benchmarks/benchmark_crop_box.py for the peak-memory/time
comparison this rewrite was built to fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.bbox3d import BBox3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import FilterError
from topocore.processing.filters.crop_box import CropBoxFilter


def _box() -> BBox3D:
    return BBox3D(min_x=0, min_y=0, min_z=0, max_x=10, max_y=10, max_z=10)


def _chunk(xs: list[float], ys: list[float], zs: list[float]) -> Chunk:
    n = len(xs)
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    return chunk


# ----------------------------------------------------------------------
# Single chunk.
# ----------------------------------------------------------------------


def test_single_chunk_points_inside_and_outside() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([5.0, 100.0, 3.0], [5.0, 100.0, 3.0], [5.0, 100.0, 3.0]))

    mask = CropBoxFilter(_box()).mask(cloud)

    np.testing.assert_array_equal(mask, [True, False, True])


# ----------------------------------------------------------------------
# Multiple chunks -- the decisive order/position check.
# ----------------------------------------------------------------------


def test_selected_point_in_a_later_chunk_not_the_first() -> None:
    """
    The decisive case: chunk 1 has NO points inside the box; only
    chunk 2's second point does. If per-chunk masks were concatenated
    out of order, or if the implementation accidentally only checked
    the first chunk, this would fail to reflect the match at its
    correct GLOBAL index (3).
    """
    cloud = PointCloud()
    cloud.add_chunk(_chunk([100.0, 200.0], [100.0, 200.0], [100.0, 200.0]))  # both outside
    cloud.add_chunk(_chunk([300.0, 5.0], [300.0, 5.0], [300.0, 5.0]))  # only index 1 (global 3) inside

    mask = CropBoxFilter(_box()).mask(cloud)

    np.testing.assert_array_equal(mask, [False, False, False, True])


def test_multiple_chunks_of_different_sizes() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 2.0, 100.0], [1.0, 2.0, 100.0], [1.0, 2.0, 100.0]))  # size 3
    cloud.add_chunk(_chunk([3.0], [3.0], [3.0]))  # size 1

    mask = CropBoxFilter(_box()).mask(cloud)

    np.testing.assert_array_equal(mask, [True, True, False, True])


def test_global_order_preserved_across_many_chunks() -> None:
    """A point's position in the returned mask must match its GLOBAL index across chunk boundaries."""
    cloud = PointCloud()
    # Alternate inside/outside across 5 single-point chunks.
    values = [5.0, 100.0, 6.0, 200.0, 7.0]
    for v in values:
        cloud.add_chunk(_chunk([v], [v], [v]))

    mask = CropBoxFilter(_box()).mask(cloud)

    np.testing.assert_array_equal(mask, [True, False, True, False, True])


# ----------------------------------------------------------------------
# Inclusive boundaries.
# ----------------------------------------------------------------------


def test_boundary_values_are_inclusive() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([0.0, 10.0, -0.001, 10.001], [0.0, 10.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]))

    mask = CropBoxFilter(_box()).mask(cloud)

    np.testing.assert_array_equal(mask, [True, True, False, False])


# ----------------------------------------------------------------------
# Empty cloud.
# ----------------------------------------------------------------------


def test_empty_cloud_rejected() -> None:
    with pytest.raises(FilterError, match="empty"):
        CropBoxFilter(_box()).mask(PointCloud())


# ----------------------------------------------------------------------
# Equivalence with the pre-PR21.7.4 concatenate-everything reference.
# ----------------------------------------------------------------------


def _reference_mask(cloud: PointCloud, box: BBox3D) -> np.ndarray:
    """The exact pre-PR21.7.4 algorithm, reimplemented here as an independent reference."""
    xs, ys, zs = [], [], []
    for chunk in cloud:
        xs.append(chunk[PointAttribute.X])
        ys.append(chunk[PointAttribute.Y])
        zs.append(chunk[PointAttribute.Z])
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    return (
        (x >= box.min_x) & (x <= box.max_x) & (y >= box.min_y) & (y <= box.max_y) & (z >= box.min_z) & (z <= box.max_z)
    )


def test_matches_reference_implementation_on_random_data() -> None:
    rng = np.random.default_rng(0)
    box = BBox3D(min_x=20, min_y=20, min_z=2, max_x=80, max_y=80, max_z=8)

    cloud = PointCloud()
    for _ in range(5):
        n = 200
        cloud.add_chunk(
            _chunk(
                list(rng.uniform(0, 100, n)),
                list(rng.uniform(0, 100, n)),
                list(rng.uniform(0, 10, n)),
            )
        )

    actual = CropBoxFilter(box).mask(cloud)
    expected = _reference_mask(cloud, box)

    np.testing.assert_array_equal(actual, expected)


def test_matches_reference_with_uneven_chunk_sizes() -> None:
    rng = np.random.default_rng(1)
    box = BBox3D(min_x=20, min_y=20, min_z=2, max_x=80, max_y=80, max_z=8)

    cloud = PointCloud()
    for n in (7, 200, 1, 53):
        cloud.add_chunk(
            _chunk(
                list(rng.uniform(0, 100, n)),
                list(rng.uniform(0, 100, n)),
                list(rng.uniform(0, 10, n)),
            )
        )

    actual = CropBoxFilter(box).mask(cloud)
    expected = _reference_mask(cloud, box)

    np.testing.assert_array_equal(actual, expected)


def test_manager_parameter_is_ignored() -> None:
    """CropBoxFilter needs no spatial structure -- passing a manager must not change the result or crash."""
    from topocore.processing.neighbors.manager import NeighborhoodManager

    cloud = PointCloud()
    cloud.add_chunk(_chunk([5.0, 100.0], [5.0, 100.0], [5.0, 100.0]))

    without_manager = CropBoxFilter(_box()).mask(cloud)
    manager = NeighborhoodManager.from_point_cloud(cloud)
    with_manager = CropBoxFilter(_box()).mask(cloud, manager=manager)

    np.testing.assert_array_equal(without_manager, with_manager)
