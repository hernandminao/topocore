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

from topocore.io.exceptions import CorruptedFileError, MissingAttributeError
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk

#: E57 fields mapped 1:1 onto a single-valued TopoCore attribute.
E57_ATTRIBUTE_MAPPING: Final = {
    "cartesianX": PointAttribute.X,
    "cartesianY": PointAttribute.Y,
    "cartesianZ": PointAttribute.Z,
    "intensity": PointAttribute.INTENSITY,
}

#: E57 stores red/green/blue as three separate fields; TopoCore
#: stores them combined as one PointAttribute.COLOR of shape (3,).
_COLOR_FIELDS = ("colorRed", "colorGreen", "colorBlue")


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

        required = set(E57_ATTRIBUTE_MAPPING) - {"intensity"}
        missing = required - scan.keys()
        if missing:
            raise MissingAttributeError(
                f"E57 scan is missing required coordinate fields: {', '.join(sorted(missing))}."
            )

        if not scan:
            raise CorruptedFileError("E57 scan contains no point data.")

        sizes = {name: np.asarray(values).shape[0] for name, values in scan.items()}
        size = next(iter(sizes.values()))
        inconsistent = {name: field_size for name, field_size in sizes.items() if field_size != size}
        if inconsistent:
            raise CorruptedFileError(
                "E57 scan arrays have inconsistent lengths: "
                + ", ".join(f"{name}={field_size}" for name, field_size in sizes.items() if field_size != size)
            )

        attributes = [attribute for name, attribute in E57_ATTRIBUTE_MAPPING.items() if name in scan]

        has_color = all(field in scan for field in _COLOR_FIELDS)

        if any(field in scan for field in _COLOR_FIELDS) and not has_color:
            raise CorruptedFileError("E57 color data must contain colorRed, colorGreen and colorBlue.")

        if has_color:
            attributes.append(PointAttribute.COLOR)

        chunk = Chunk(
            size=size,
            attributes=attributes,
            source_id=source_id,
        )

        for name, attribute in E57_ATTRIBUTE_MAPPING.items():
            if name in scan:
                chunk[attribute][:] = np.asarray(scan[name])

        if has_color:
            chunk[PointAttribute.COLOR][:] = np.stack(
                [np.asarray(scan[field]) for field in _COLOR_FIELDS],
                axis=1,
            )

        return chunk


__all__ = [
    "E57Converter",
    "E57_ATTRIBUTE_MAPPING",
]
