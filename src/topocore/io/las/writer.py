"""
topocore.io.las.writer
======================

Writer for ASPRS LAS files.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path

import laspy  # type: ignore[import-untyped]
import numpy as np

from topocore.io.base import PointCloudWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud

#: ASPRS-recommended default scale factor (1mm) -- finer than
#: laspy's own internal default (0.01, 1cm). Found and fixed in
#: PR19: LASWriter previously never set header.scales/header.offsets
#: at all, silently relying on laspy's coarser 1cm default for EVERY
#: write, regardless of the actual precision of the source data.
#: Confirmed directly with a write->read round trip on realistic
#: UTM-style survey coordinates: 500123.456 became 500123.46 --
#: exactly the kind of silent precision loss that matters for
#: GNSS RTK-grade survey workflows (millimeter precision), which
#: this library is explicitly built around.
_DEFAULT_SCALE = (0.001, 0.001, 0.001)


class LASWriter(PointCloudWriter):
    """
    Writer for ASPRS LAS files.

    Parameters
    ----------
    path
        Destination file.
    point_format
        LAS point format.
    version
        LAS version string.
    scale
        Per-axis (x, y, z) scale factor used to encode coordinates
        as scaled integers. If not given, defaults to 1mm on each
        axis (the ASPRS-recommended default) -- NOT laspy's own,
        coarser 1cm internal default.
    offset
        Per-axis (x, y, z) offset. If not given, it is computed
        automatically from the minimum coordinate of the data being
        written, keeping the internal scaled-integer values small
        and avoiding unnecessary precision loss far from the
        coordinate origin.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        point_format: int = 3,
        version: str = "1.2",
        scale: tuple[float, float, float] | None = None,
        offset: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__(path)

        self._point_format = point_format
        self._version = version
        self._scale = scale
        self._offset = offset

    @property
    def point_format(self) -> int:
        """LAS point format."""
        return self._point_format

    @property
    def version(self) -> str:
        """LAS version."""
        return self._version

    @property
    def scale(self) -> tuple[float, float, float] | None:
        """Explicitly configured scale, if any."""
        return self._scale

    @property
    def offset(self) -> tuple[float, float, float] | None:
        """Explicitly configured offset, if any."""
        return self._offset

    def write(
        self,
        cloud: PointCloud,
    ) -> None:
        """
        Write a PointCloud into a LAS file.
        """

        arrays: dict[PointAttribute, list[np.ndarray]] = {}

        for chunk in cloud:
            for attribute in chunk.attributes:
                arrays.setdefault(attribute, []).append(chunk[attribute])

        merged = {attribute: np.concatenate(values) for attribute, values in arrays.items()}

        header = laspy.LasHeader(
            point_format=self._point_format,
            version=self._version,
        )

        header.scales = list(self._scale) if self._scale is not None else list(_DEFAULT_SCALE)

        if self._offset is not None:
            header.offsets = list(self._offset)
        elif PointAttribute.X in merged and PointAttribute.Y in merged and PointAttribute.Z in merged:
            header.offsets = [
                float(np.min(merged[PointAttribute.X])),
                float(np.min(merged[PointAttribute.Y])),
                float(np.min(merged[PointAttribute.Z])),
            ]

        las = laspy.LasData(header)

        if PointAttribute.X in merged:
            las.x = merged[PointAttribute.X]

        if PointAttribute.Y in merged:
            las.y = merged[PointAttribute.Y]

        if PointAttribute.Z in merged:
            las.z = merged[PointAttribute.Z]

        if PointAttribute.INTENSITY in merged:
            las.intensity = merged[PointAttribute.INTENSITY]

        if PointAttribute.CLASSIFICATION in merged:
            las.classification = merged[PointAttribute.CLASSIFICATION]

        if PointAttribute.RETURN_NUMBER in merged:
            las.return_number = merged[PointAttribute.RETURN_NUMBER]

        if PointAttribute.NUMBER_OF_RETURNS in merged:
            las.number_of_returns = merged[PointAttribute.NUMBER_OF_RETURNS]

        if PointAttribute.GPS_TIME in merged:
            las.gps_time = merged[PointAttribute.GPS_TIME]

        # PointAttribute.COLOR is stored combined, shape (n, 3); LAS
        # itself keeps red/green/blue as three separate channels, so
        # it's split back out here on the way out.
        if PointAttribute.COLOR in merged:
            color = merged[PointAttribute.COLOR]
            las.red = color[:, 0]
            las.green = color[:, 1]
            las.blue = color[:, 2]

        las.write(self.path)

    def close(self) -> None:
        """
        Release writer resources.

        No persistent resources are kept by LASWriter.
        """


__all__ = [
    "LASWriter",
]
