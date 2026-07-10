"""
topocore.io.e57.converter
=========================

Conversion utilities between E57 scan data and TopoCore objects.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk

#: Mapping between E57 field names and TopoCore point attributes.
E57_ATTRIBUTE_MAPPING: Final = {
    "cartesianX": PointAttribute.X,
    "cartesianY": PointAttribute.Y,
    "cartesianZ": PointAttribute.Z,
    "intensity": PointAttribute.INTENSITY,
    "colorRed": PointAttribute.RED,
    "colorGreen": PointAttribute.GREEN,
    "colorBlue": PointAttribute.BLUE,
}


class E57Converter:
    """
    Converts E57 scan data into TopoCore chunks.
    """

    @staticmethod
    def from_scan(
        scan: dict[str, np.ndarray],
        *,
        source_id: int,
    ) -> Chunk:
        """
        Convert one E57 scan slice into a Chunk.
        """

        attributes: list[PointAttribute] = []

        for name, attribute in E57_ATTRIBUTE_MAPPING.items():
            if name in scan:
                attributes.append(attribute)

        size = len(next(iter(scan.values())))

        chunk = Chunk(
            size=size,
            attributes=attributes,
            source_id=source_id,
        )

        for name, attribute in E57_ATTRIBUTE_MAPPING.items():
            if name in scan:
                chunk[attribute][:] = np.asarray(scan[name])

        return chunk
