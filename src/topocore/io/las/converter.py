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

from typing import Any

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk

#: LAS fields mapped 1:1 onto a single-valued TopoCore attribute.
_SCALAR_MAPPING = {
    "x": PointAttribute.X,
    "y": PointAttribute.Y,
    "z": PointAttribute.Z,
    "intensity": PointAttribute.INTENSITY,
    "classification": PointAttribute.CLASSIFICATION,
    "return_number": PointAttribute.RETURN_NUMBER,
    "number_of_returns": PointAttribute.NUMBER_OF_RETURNS,
    "gps_time": PointAttribute.GPS_TIME,
}

#: LAS stores red/green/blue as three separate fields; TopoCore
#: stores them combined as one PointAttribute.COLOR of shape (3,).
_COLOR_FIELDS = ("red", "green", "blue")


class LASConverter:
    """
    Converts LAS point records into TopoCore chunks.
    """

    @staticmethod
    def from_las_points(points: Any) -> Chunk:
        """
        Convert a laspy PackedPointRecord into a Chunk.

        Parameters
        ----------
        points
            Packed point record returned by laspy. Typed as ``Any``:
            laspy ships no type stubs / py.typed marker, so this is
            the accurate type for an object from an untyped library,
            not a placeholder for "didn't bother."

        Returns
        -------
        Chunk
        """

        available = [attribute for las_name, attribute in _SCALAR_MAPPING.items() if hasattr(points, las_name)]

        has_color = all(hasattr(points, field) for field in _COLOR_FIELDS)

        if has_color:
            available.append(PointAttribute.COLOR)

        chunk = Chunk(
            size=len(points),
            attributes=available,
            source_id=0,
        )

        for las_name, attribute in _SCALAR_MAPPING.items():
            if hasattr(points, las_name):
                chunk[attribute][:] = np.asarray(getattr(points, las_name))

        if has_color:
            chunk[PointAttribute.COLOR][:] = np.stack(
                [np.asarray(getattr(points, field)) for field in _COLOR_FIELDS],
                axis=1,
            )

        return chunk


__all__ = [
    "LASConverter",
]
