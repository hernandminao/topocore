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
from topocore.io.exceptions import MissingAttributeError, WriteError
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud


class LASWriter(PointCloudWriter):
    """
    Writer for ASPRS LAS files.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        point_format: int = 3,
        version: str = "1.2",
    ) -> None:
        super().__init__(path)

        self._point_format = point_format
        self._version = version

    @property
    def point_format(self) -> int:
        """LAS point format."""
        return self._point_format

    @property
    def version(self) -> str:
        """LAS version."""
        return self._version

    def write(
        self,
        cloud: PointCloud,
    ) -> None:
        """
        Write a PointCloud into a LAS file.
        """

        header = laspy.LasHeader(
            point_format=self._point_format,
            version=self._version,
        )

        las = laspy.LasData(header)

        arrays: dict[PointAttribute, list[np.ndarray]] = {}

        for chunk in cloud:
            for attribute in chunk.attributes:
                arrays.setdefault(attribute, []).append(chunk[attribute])

        merged = {attribute: np.concatenate(values) for attribute, values in arrays.items()}

        required = (
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
        )
        missing = [attribute.name for attribute in required if attribute not in merged]
        if missing:
            raise MissingAttributeError(f"LAS output requires X, Y and Z attributes; missing: {', '.join(missing)}.")

        point_count = merged[PointAttribute.X].shape[0]
        for attribute, values in merged.items():
            if values.shape[0] != point_count:
                raise WriteError(
                    f"Attribute {attribute.name} contains {values.shape[0]} values, expected {point_count}."
                )

        if PointAttribute.COLOR in merged:
            color = np.asarray(merged[PointAttribute.COLOR])
            if color.ndim != 2 or color.shape != (point_count, 3):
                raise WriteError("LAS COLOR attribute must have shape (point_count, 3).")

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

        try:
            las.write(self.path)
        except Exception as exc:
            raise WriteError(f"Unable to write LAS file '{self.path}'.") from exc

    def close(self) -> None:
        """
        Release writer resources.

        No persistent resources are kept by LASWriter.
        """
        pass


__all__ = [
    "LASWriter",
]
