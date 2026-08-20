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

from topocore.pointcloud.attributes import ATTRIBUTE_DTYPES, PointAttribute
from topocore.pointcloud.chunk import Chunk

#: E57 fields mapped 1:1 onto a single-valued TopoCore attribute.
#: "intensity" is handled separately -- see _scale_intensity below.
E57_ATTRIBUTE_MAPPING: Final = {
    "cartesianX": PointAttribute.X,
    "cartesianY": PointAttribute.Y,
    "cartesianZ": PointAttribute.Z,
}

#: E57 stores red/green/blue as three separate fields; TopoCore
#: stores them combined as one PointAttribute.COLOR of shape (3,).
_COLOR_FIELDS = ("colorRed", "colorGreen", "colorBlue")


def _scale_intensity(raw: np.ndarray) -> np.ndarray:
    """
    Rescale raw E57 intensity into TopoCore's unified uint16 range.

    Found and fixed in PR19: E57's "intensity" field has no fixed,
    universal range -- pye57's own read_scan() returns whatever raw
    values the file stores, unscaled (confirmed by reading pye57's
    own source: it applies no normalization at all). Per the ASTM
    E57 spec, a file MAY declare its own intensityLimits, but common
    real-world files often use a normalized [0, 1] float range.
    PointAttribute.INTENSITY, however, is a unified,
    format-agnostic uint16 field (matching LAS's own raw-count
    convention). Directly assigning E57's float intensity into that
    uint16 column (as the previous code did, since "intensity"
    wasn't even being requested from pye57 at all -- a separate bug
    also fixed in this session) silently truncated every value to 0
    for the common [0, 1]-normalized case: confirmed directly with a
    real E57 file containing intensity=[0.1, 0.5, 0.9], which became
    [0, 0, 0].

    Fixed with data-driven min-max normalization: rescale the
    ACTUAL observed range in this scan to fill [0, 65535], rather
    than assuming any fixed a-priori range. This correctly handles
    both already-[0,1]-normalized files and files using a different
    native range, without needing to parse E57's own
    intensityLimits metadata.
    """

    raw = np.asarray(raw, dtype=np.float64)

    minimum = float(np.min(raw))
    maximum = float(np.max(raw))

    if maximum <= minimum:
        # Degenerate case: every value identical (including a
        # single-point scan) -- map to the middle of the range
        # rather than dividing by zero.
        return np.full(raw.shape, 32767, dtype=ATTRIBUTE_DTYPES[PointAttribute.INTENSITY])

    scaled = (raw - minimum) / (maximum - minimum) * 65535.0

    return scaled.astype(ATTRIBUTE_DTYPES[PointAttribute.INTENSITY])


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

        attributes = [attribute for name, attribute in E57_ATTRIBUTE_MAPPING.items() if name in scan]

        has_intensity = "intensity" in scan

        if has_intensity:
            attributes.append(PointAttribute.INTENSITY)

        has_color = all(field in scan for field in _COLOR_FIELDS)

        if has_color:
            attributes.append(PointAttribute.COLOR)

        size = len(next(iter(scan.values())))

        chunk = Chunk(
            size=size,
            attributes=attributes,
            source_id=source_id,
        )

        for name, attribute in E57_ATTRIBUTE_MAPPING.items():
            if name in scan:
                chunk[attribute][:] = np.asarray(scan[name])

        if has_intensity:
            chunk[PointAttribute.INTENSITY][:] = _scale_intensity(scan["intensity"])

        if has_color:
            chunk[PointAttribute.COLOR][:] = np.stack(
                [np.asarray(scan[field]) for field in _COLOR_FIELDS],
                axis=1,
            )

        return chunk


__all__ = [
    "E57_ATTRIBUTE_MAPPING",
    "E57Converter",
]
