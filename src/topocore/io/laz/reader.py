"""
topocore.io.laz.reader
======================

Reader for compressed ASPRS LAZ files.

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
from topocore.io.exceptions import PointCloudIOError
from topocore.io.las.base_reader import BaseLASReader
from topocore.pointcloud.chunk import Chunk


class LAZReader(BaseLASReader):
    """
    Reader for compressed ASPRS LAZ files.
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
        Open the LAZ file.
        """
        if self._reader is not None:
            return

        try:
            self._reader = laspy.open(self.path)

        except FileNotFoundError as exc:
            raise PointCloudIOError(f"File not found: {self.path}") from exc

        except Exception as exc:
            raise PointCloudIOError(
                "Unable to open the LAZ file. "
                "Ensure the file is valid and the LAZ backend "
                "(lazrs) is correctly installed."
            ) from exc

    def _iterate_chunks(self) -> Iterator[Chunk]:
        """
        Iterate over LAZ chunks.
        """
        if self._reader is None:
            self._open()

        assert self._reader is not None

        from topocore.io.las.converter import LASConverter

        for points in self._reader.chunk_iterator(self._chunk_size):
            yield LASConverter.from_las_points(points)


__all__ = [
    "LAZReader",
]
