"""
topocore.io.las.reader
======================

Reader for ASPRS LAS files.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import laspy  # type: ignore[import-untyped]

from topocore.io.constants import DEFAULT_CHUNK_SIZE
from topocore.io.crs.external import apply_crs_with_native_priority
from topocore.io.exceptions import PointCloudIOError
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud

from .base_reader import BaseLASReader
from .converter import LASConverter
from .crs_detection import detect_crs


class LASReader(BaseLASReader):
    """
    Reader for ASPRS LAS files.

    The reader streams the file using laspy's chunk iterator and converts
    each block into a TopoCore Chunk.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(path)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        self._chunk_size = chunk_size

    @property
    def chunk_size(self) -> int:
        """
        Number of points read per iteration.
        """
        return self._chunk_size

    def _open(self) -> None:
        """
        Open the LAS file.
        """
        if self._reader is not None:
            return

        try:
            self._reader = laspy.open(self.path)

        except FileNotFoundError as exc:
            raise PointCloudIOError(f"File not found: {self.path}") from exc

        except Exception as exc:
            raise PointCloudIOError(f"Unable to open LAS file '{self.path}'.") from exc

    def _iterate_chunks(self) -> Iterator[Chunk]:
        """
        Iterate over LAS chunks.
        """
        if self._reader is None:
            self._open()

        assert self._reader is not None

        try:
            for points in self._reader.chunk_iterator(self._chunk_size):
                yield LASConverter.from_las_points(points)
        except Exception as exc:
            raise PointCloudIOError(f"Unable to read point data from '{self.path}'.") from exc

    def read(self) -> PointCloud:
        """
        Same base contract as `PointCloudReader.read()` -- no new
        parameter, no signature change. The one addition: a native
        CRS embedded in this LAS/LAZ file's own header (see
        `topocore.io.las.crs_detection.detect_crs()`) always takes
        priority; only when the file declares no native CRS is a
        `.prj` sidecar consulted as a fallback (see
        `topocore.io.crs.ExternalCRSDetector`). A `.prj` sidecar
        NEVER overwrites a valid native CRS, even one declaring a
        genuinely different value -- confirmed directly this holds
        regardless of whether the two agree. Coordinates (`X`/`Y`/`Z`)
        are never touched by this -- only `PointCloud.crs` is
        affected; this is detection/annotation, never a
        transformation.
        """
        cloud = super().read()
        native_crs = detect_crs(self.path)
        return apply_crs_with_native_priority(cloud, self.path, native_crs)


__all__ = [
    "LASReader",
]
