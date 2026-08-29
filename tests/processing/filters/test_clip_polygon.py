"""
Coverage audit tests for topocore.processing.filters.clip_polygon
(point_in_polygon() and ClipPolygonFilter).

point_in_polygon() is a public, standalone function (in __all__),
separate from ClipPolygonFilter. Its own "at least three vertices"
check is confirmed genuinely reachable by calling it directly with a
malformed polygon -- even though ClipPolygonFilter.__init__ already
validates this before ever reaching point_in_polygon() through the
class's own mask() path, nothing prevents an external caller from
using point_in_polygon() on its own, matching the same reasoning
already established for build_cloud()'s own audit.

mask()'s broad `except Exception` wrapping (unlike pass_through.py's
narrower `except ProcessingError`) is confirmed reachable via
heterogeneous chunks, with __cause__ preserved as ProcessingError.

"Point cloud must contain X/Y coordinates" is confirmed unreachable:
Chunk.__init__ already requires X, Y, and Z at construction, so no
PointCloud can ever lack X or Y.

name() and the polygon property are documented as orphaned -- zero
external callers confirmed via grep.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import FilterError, ProcessingError
from topocore.processing.filters.clip_polygon import ClipPolygonFilter, point_in_polygon


def _square_cloud() -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [5.0, 50.0]
    chunk[PointAttribute.Y][:] = [5.0, 50.0]
    chunk[PointAttribute.Z][:] = [0.0, 0.0]
    cloud.add_chunk(chunk)
    return cloud


_SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


# ----------------------------------------------------------------------
# point_in_polygon() -- its own directly-callable public contract.
# ----------------------------------------------------------------------


def test_point_in_polygon_rejects_fewer_than_three_vertices_directly() -> None:
    with pytest.raises(FilterError, match="at least three vertices"):
        point_in_polygon(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )


def test_point_in_polygon_happy_path() -> None:
    result = point_in_polygon(
        np.array([5.0, 50.0]),
        np.array([5.0, 50.0]),
        _SQUARE[:, 0],
        _SQUARE[:, 1],
    )
    np.testing.assert_array_equal(result, [True, False])


# ----------------------------------------------------------------------
# ClipPolygonFilter.__init__ -- validation and polygon reshaping.
# ----------------------------------------------------------------------


def test_wrong_polygon_shape_rejected() -> None:
    with pytest.raises(FilterError, match=r"shape \(M, 2\)"):
        ClipPolygonFilter(np.array([[0.0, 0.0, 0.0]]))


def test_fewer_than_three_vertices_rejected_at_construction() -> None:
    with pytest.raises(FilterError, match="at least 3 vertices"):
        ClipPolygonFilter(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_flat_1d_polygon_is_reshaped_to_mx2() -> None:
    flat_polygon = np.array([0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0])
    f = ClipPolygonFilter(flat_polygon)
    assert f.polygon.shape == (4, 2)


# ----------------------------------------------------------------------
# mask() -- empty cloud, happy path, exception wrapping.
# ----------------------------------------------------------------------


def test_mask_rejects_empty_cloud() -> None:
    with pytest.raises(FilterError, match="empty point cloud"):
        ClipPolygonFilter(_SQUARE).mask(PointCloud())


def test_mask_happy_path() -> None:
    mask = ClipPolygonFilter(_SQUARE).mask(_square_cloud())
    np.testing.assert_array_equal(mask, [True, False])


def test_mask_wraps_broad_exception_from_heterogeneous_chunks() -> None:
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
        ClipPolygonFilter(_SQUARE).mask(cloud)

    assert isinstance(exc_info.value.__cause__, ProcessingError)
