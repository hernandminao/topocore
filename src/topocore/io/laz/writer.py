"""
topocore.io.laz.writer
======================

Writer for compressed ASPRS LAZ files.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from topocore.io.base import PointCloudWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud


class LAZWriter(PointCloudWriter):
    """
    Writer for compressed ASPRS LAZ files.
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

    def write(
        self,
        cloud: PointCloud,
    ) -> None:
        """
        Write a PointCloud into a compressed LAZ file.
        """

        header = laspy.LasHeader(
            point_format=self._point_format,
            version=self._version,
        )

        las = laspy.LasData(header)

        arrays: dict[PointAttribute, list[np.ndarray]] = {}

        for chunk in cloud:
            for attribute in chunk.attributes:
                arrays.setdefault(attribute, []).append(
                    chunk[attribute]
                )

        merged = {
            attribute: np.concatenate(values)
            for attribute, values in arrays.items()
        }

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
            las.number_of_returns = merged[
                PointAttribute.NUMBER_OF_RETURNS
            ]

        if PointAttribute.GPS_TIME in merged:
            las.gps_time = merged[PointAttribute.GPS_TIME]

        if PointAttribute.RED in merged:
            las.red = merged[PointAttribute.RED]

        if PointAttribute.GREEN in merged:
            las.green = merged[PointAttribute.GREEN]

        if PointAttribute.BLUE in merged:
            las.blue = merged[PointAttribute.BLUE]

        las.write(
            self.path,
            do_compress=True,
        )

    def close(self) -> None:
        """
        Release writer resources.

        This implementation is intentionally a no-op because no
        persistent resources are held.
        """
        pass