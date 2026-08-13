"""
topocore.terrain.conversion
===============================

Adapts `topocore.pointcloud.PointCloud` into the
``tuple[Point3D, ...]`` that `TIN.from_points()` (and other
Point3D-based Terrain consumers -- profiles, breaklines, visibility)
expect.

This conversion is deliberately NOT owned by any orchestrator
(Workflow or otherwise) -- it's domain logic that belongs to
Terrain, reusable independently of how the caller obtained the
PointCloud, and not tied to any particular pipeline.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geometry.point3d import Point3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.terrain.exceptions import ConversionError


def pointcloud_to_points(cloud: PointCloud) -> tuple[Point3D, ...]:
    """
    Convert every point in ``cloud`` into a ``Point3D``, preserving
    order (chunk order, then within-chunk order).

    Parameters
    ----------
    cloud
        Source point cloud. Every chunk is guaranteed to have X, Y,
        and Z -- as of TD-004, `Chunk.__init__` itself rejects
        construction if any of them is missing (they're declared
        ``REQUIRED`` in ``pointcloud.attributes.ATTRIBUTE_DEFINITIONS``),
        so there is no longer a per-chunk check to make here.

    Raises
    ------
    ConversionError
        If ``cloud`` is empty.
    """
    if cloud.is_empty:
        raise ConversionError("Cannot convert an empty PointCloud to Point3D.")

    points: list[Point3D] = []

    for chunk in cloud:
        xs = chunk[PointAttribute.X]
        ys = chunk[PointAttribute.Y]
        zs = chunk[PointAttribute.Z]

        points.extend(Point3D(float(x), float(y), float(z)) for x, y, z in zip(xs, ys, zs, strict=True))

    return tuple(points)


__all__ = ["pointcloud_to_points"]
