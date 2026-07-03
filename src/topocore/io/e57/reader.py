"""
topocore.io.e57.reader
======================

Reader for ASTM E57 point cloud files.

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

import numpy as np
import pye57

from topocore.io.exceptions import PointCloudIOError
from topocore.pointcloud.chunk import Chunk

from .base_reader import BaseE57Reader
from .converter import E57Converter


class E57Reader(BaseE57Reader):
    """
    Reader for ASTM E57 files.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int,
    ) -> None:
        super().__init__(
            path,
            chunk_size=chunk_size,
        )

    def _open(self) -> None:

        if self._reader is not None:
            return

        try:
            self._reader = pye57.E57(str(self.path))

        except Exception as exc:
            raise PointCloudIOError(
                f"Unable to open E57 file '{self.path}'."
            ) from exc

    def _iterate_scans(
        self,
    ) -> Iterator[
        tuple[int, dict[str, np.ndarray]]
    ]:

        assert self._reader is not None

        for scan_index in range(
            self._reader.scan_count
        ):

            yield (
                scan_index,
                self._reader.read_scan(scan_index),
            )

    def _create_chunk(
        self,
        arrays: dict[str, np.ndarray],
        source_id: int,
    ) -> Chunk:

        return E57Converter.from_scan(
            arrays,
            source_id=source_id,
        )