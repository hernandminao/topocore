"""
Regression suite for topocore.processing.filters.crop_box,
.pass_through, and .clip_polygon -- PR19.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.bbox3d import BBox3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import FilterError
from topocore.processing.filters.clip_polygon import ClipPolygonFilter, point_in_polygon
from topocore.processing.filters.crop_box import CropBoxFilter
from topocore.processing.filters.pass_through import Axis, PassThroughFilter


def _cloud(xs: list[float], ys: list[float], zs: list[float]) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# CropBoxFilter
# ----------------------------------------------------------------------


def test_crop_box_keeps_points_inside_all_three_axes() -> None:
    box = BBox3D(min_x=0, min_y=0, min_z=0, max_x=10, max_y=10, max_z=10)
    cloud = _cloud([5.0, 15.0, 5.0, 5.0], [5.0, 5.0, 15.0, 5.0], [5.0, 5.0, 5.0, 15.0])
    mask = CropBoxFilter(box).mask(cloud)
    np.testing.assert_array_equal(mask, [True, False, False, False])


def test_crop_box_boundary_inclusive() -> None:
    box = BBox3D(min_x=0, min_y=0, min_z=0, max_x=10, max_y=10, max_z=10)
    cloud = _cloud([0.0, 10.0], [0.0, 10.0], [0.0, 10.0])
    mask = CropBoxFilter(box).mask(cloud)
    assert mask.all()


def test_crop_box_rejects_empty_cloud() -> None:
    box = BBox3D(min_x=0, min_y=0, min_z=0, max_x=1, max_y=1, max_z=1)
    with pytest.raises(FilterError):
        CropBoxFilter(box).mask(PointCloud())


# ----------------------------------------------------------------------
# PassThroughFilter
# ----------------------------------------------------------------------


def test_pass_through_z_axis() -> None:
    cloud = _cloud([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 50.0, -5.0])
    mask = PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0).mask(cloud)
    np.testing.assert_array_equal(mask, [True, False, False])


def test_pass_through_boundary_inclusive() -> None:
    cloud = _cloud([0.0], [0.0], [10.0])
    mask = PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0).mask(cloud)
    assert mask[0]


def test_pass_through_rejects_min_greater_than_max() -> None:
    with pytest.raises(FilterError):
        PassThroughFilter(Axis.X, min_value=10.0, max_value=0.0)


def test_pass_through_rejects_nan_values() -> None:
    cloud = _cloud([0.0, float("nan")], [0.0, 0.0], [0.0, 0.0])
    with pytest.raises(FilterError):
        PassThroughFilter(Axis.X, min_value=-1.0, max_value=1.0).mask(cloud)


# ----------------------------------------------------------------------
# point_in_polygon / ClipPolygonFilter
# ----------------------------------------------------------------------


def test_point_in_polygon_simple_square() -> None:
    poly_x = np.array([0.0, 10.0, 10.0, 0.0])
    poly_y = np.array([0.0, 0.0, 10.0, 10.0])
    x = np.array([5.0, 15.0, -5.0, 5.0])
    y = np.array([5.0, 15.0, 5.0, -5.0])

    result = point_in_polygon(x, y, poly_x, poly_y)
    np.testing.assert_array_equal(result, [True, False, False, False])


def test_point_in_polygon_concave_l_shape() -> None:
    """
    A concave L-shaped polygon (outer square minus a notch) --
    ray casting must correctly exclude the notch, not just handle
    convex shapes.
    """
    poly_x = np.array([0.0, 10.0, 10.0, 5.0, 5.0, 0.0])
    poly_y = np.array([0.0, 0.0, 5.0, 5.0, 10.0, 10.0])

    x = np.array([7.0, 2.0, 8.0, 3.0])
    y = np.array([7.0, 2.0, 2.0, 8.0])

    result = point_in_polygon(x, y, poly_x, poly_y)
    np.testing.assert_array_equal(result, [False, True, True, True])  # (7,7) is in the notch


def test_point_in_polygon_rejects_fewer_than_three_vertices() -> None:
    with pytest.raises(FilterError):
        point_in_polygon(np.array([0.0]), np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_clip_polygon_filter_matches_point_in_polygon() -> None:
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    cloud = _cloud([5.0, 15.0], [5.0, 15.0], [0.0, 0.0])

    mask = ClipPolygonFilter(poly).mask(cloud)
    np.testing.assert_array_equal(mask, [True, False])


def test_clip_polygon_rejects_fewer_than_three_vertices() -> None:
    with pytest.raises(FilterError):
        ClipPolygonFilter(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_clip_polygon_rejects_wrong_shape() -> None:
    with pytest.raises(FilterError):
        ClipPolygonFilter(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
