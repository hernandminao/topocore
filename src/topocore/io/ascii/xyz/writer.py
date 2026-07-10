"""
topocore.io.ascii.xyz.writer
============================

Writer for XYZ ASCII point cloud files.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.io.ascii.base_writer import BaseASCIIWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud


class XYZWriter(BaseASCIIWriter):
    """
    Writer for XYZ ASCII files.
    """

    def write(
        self,
        point_cloud: PointCloud,
    ) -> None:

        with self.path.open(
            "w",
            encoding=self.encoding,
        ) as stream:
            for chunk in point_cloud:
                x = chunk[PointAttribute.X]
                y = chunk[PointAttribute.Y]
                z = chunk[PointAttribute.Z]

                for xi, yi, zi in zip(
                    x,
                    y,
                    z,
                    strict=True,
                ):
                    stream.write(f"{xi} {yi} {zi}\n")
