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

        PR21.7.3: writes each Chunk incrementally via laspy's
        low-level LasWriter.write_points() instead of first merging
        every chunk's attribute arrays into one giant array per
        attribute via np.concatenate(). Confirmed via PR21.7.2's own
        peak-RSS benchmark (benchmarks/benchmark_las_writer_memory.py)
        that the old merge step's memory overhead grows worse than
        linearly relative to the underlying chunk data as point count
        grows (1.4x at 100,000 points, 2.8x at 5,000,000) -- this
        eliminates that overhead entirely while producing numerically
        and semantically identical output (verified in this PR's own
        regression suite via a full write -> read -> compare
        round trip, matching a merged-array reference write exactly
        on point_count, point_format, version, scales, offsets, and
        every attribute's values and order).

        The only case genuinely requiring a pass over chunk data
        before writing begins is computing an automatic offset (when
        `offset` is not explicitly given): this is done as a cheap
        min/min/min reduction directly on each chunk's own X/Y/Z
        arrays (never concatenated, never copied) rather than the
        prior single global np.min() over a merged array -- when an
        explicit `offset` is provided, no such pass happens at all.
        """
        if cloud.is_empty:
            header = laspy.LasHeader(
                point_format=self._point_format,
                version=self._version,
            )
            header.scales = list(self._scale) if self._scale is not None else list(_DEFAULT_SCALE)
            if self._offset is not None:
                header.offsets = list(self._offset)
            laspy.LasData(header).write(self.path)
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
            las_writer = laspy.LasWriter(file_object, header)

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

        No persistent resources are kept by LASWriter.
        """


__all__ = [
    "LASWriter",
]
