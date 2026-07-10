"""
topocore.io.las.converter
=========================

Conversion utilities between LAS point records and TopoCore objects.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk


class LASConverter:
    """
    Converts LAS point records into TopoCore chunks.
    """

    @staticmethod
    def from_las_points(points) -> Chunk:
        """
        Convert a laspy PackedPointRecord into a Chunk.

        Parameters
        ----------
        points
            Packed point record returned by laspy.

        Returns
        -------
        Chunk
        """

        available = []

        mapping = {
            "x": PointAttribute.X,
            "y": PointAttribute.Y,
            "z": PointAttribute.Z,
            "intensity": PointAttribute.INTENSITY,
            "classification": PointAttribute.CLASSIFICATION,
            "return_number": PointAttribute.RETURN_NUMBER,
            "number_of_returns": PointAttribute.NUMBER_OF_RETURNS,
            "gps_time": PointAttribute.GPS_TIME,
            "red": PointAttribute.RED,
            "green": PointAttribute.GREEN,
            "blue": PointAttribute.BLUE,
        }

        for las_name, attribute in mapping.items():
            if hasattr(points, las_name):
                available.append(attribute)

        chunk = Chunk(
            size=len(points),
            attributes=available,
            source_id=0,
        )

        for las_name, attribute in mapping.items():
            if hasattr(points, las_name):
                chunk[attribute][:] = np.asarray(getattr(points, las_name))

        return chunk
