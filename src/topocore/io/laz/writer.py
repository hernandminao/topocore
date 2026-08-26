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

import laspy  # type: ignore[import-untyped]
import numpy as np

from topocore.io.base import PointCloudWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud

#: ASPRS-recommended default scale factor (1mm) -- finer than
#: laspy's own internal default (0.01, 1cm). Found and fixed in
#: PR19: this writer -- a near-duplicate of LASWriter, sharing the
#: exact same bug -- never set header.scales/header.offsets at all,
#: silently relying on laspy's coarser 1cm default for EVERY write.
#: Confirmed directly with a write->read round trip on realistic
#: UTM-style survey coordinates through a real compressed .laz file
#: (lazrs backend): 500123.456 became 500123.46. See
#: topocore.io.las.writer for the identical original finding.
_DEFAULT_SCALE = (0.001, 0.001, 0.001)


class LAZWriter(PointCloudWriter):
    """
    Writer for compressed ASPRS LAZ files.

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
        written.
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
        Write a PointCloud into a compressed LAZ file.

        PR21.7.3: mirrors the identical fix applied to
        topocore.io.las.writer.LASWriter -- see that class's own
        `write()` docstring for the full rationale (this was, and
        remains, a near-duplicate of that writer, sharing the exact
        same PR19 scale/offset finding and now the same PR21.7.3
        streaming fix).
        """
        if cloud.is_empty:
            header = laspy.LasHeader(
                point_format=self._point_format,
                version=self._version,
            )
            header.scales = list(self._scale) if self._scale is not None else list(_DEFAULT_SCALE)
            if self._offset is not None:
                header.offsets = list(self._offset)
            laspy.LasData(header).write(self.path, do_compress=True)
            return

        first_chunk = next(iter(cloud))
        attributes = first_chunk.attributes

        header = laspy.LasHeader(
            point_format=self._point_format,
            version=self._version,
        )
        header.scales = list(self._scale) if self._scale is not None else list(_DEFAULT_SCALE)

        if self._offset is not None:
            header.offsets = list(self._offset)
        elif PointAttribute.X in attributes and PointAttribute.Y in attributes and PointAttribute.Z in attributes:
            min_x = min_y = min_z = float("inf")
            for chunk in cloud:
                min_x = min(min_x, float(np.min(chunk[PointAttribute.X])))
                min_y = min(min_y, float(np.min(chunk[PointAttribute.Y])))
                min_z = min(min_z, float(np.min(chunk[PointAttribute.Z])))
            header.offsets = [min_x, min_y, min_z]

        with open(self.path, "wb") as file_object:
            las_writer = laspy.LasWriter(file_object, header, do_compress=True)

            try:
                for chunk in cloud:
                    record = laspy.ScaleAwarePointRecord.zeros(chunk.size, header=las_writer.header)

                    if PointAttribute.X in attributes:
                        record.x = chunk[PointAttribute.X]

                    if PointAttribute.Y in attributes:
                        record.y = chunk[PointAttribute.Y]

                    if PointAttribute.Z in attributes:
                        record.z = chunk[PointAttribute.Z]

                    if PointAttribute.INTENSITY in attributes:
                        record.intensity = chunk[PointAttribute.INTENSITY]

                    if PointAttribute.CLASSIFICATION in attributes:
                        record.classification = chunk[PointAttribute.CLASSIFICATION]

                    if PointAttribute.RETURN_NUMBER in attributes:
                        record.return_number = chunk[PointAttribute.RETURN_NUMBER]

                    if PointAttribute.NUMBER_OF_RETURNS in attributes:
                        record.number_of_returns = chunk[PointAttribute.NUMBER_OF_RETURNS]

                    if PointAttribute.GPS_TIME in attributes:
                        record.gps_time = chunk[PointAttribute.GPS_TIME]

                    # PointAttribute.COLOR is stored combined, shape (n, 3); LAS
                    # itself keeps red/green/blue as three separate channels, so
                    # it's split back out here on the way out.
                    if PointAttribute.COLOR in attributes:
                        color = chunk[PointAttribute.COLOR]
                        record.red = color[:, 0]
                        record.green = color[:, 1]
                        record.blue = color[:, 2]

                    las_writer.write_points(record)
            finally:
                las_writer.close()

    def close(self) -> None:
        """
        Release writer resources.

        This implementation is intentionally a no-op because no
        persistent resources are held.
        """


__all__ = [
    "LAZWriter",
]
