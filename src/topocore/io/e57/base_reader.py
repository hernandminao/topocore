"""
topocore.io.e57.base_reader
===========================

Base implementation shared by E57 readers.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator

import numpy as np

from topocore.io.base import PointCloudReader
from topocore.io.constants import DEFAULT_CHUNK_SIZE
from topocore.pointcloud.chunk import Chunk


class BaseE57Reader(PointCloudReader):
    """
    Base class for E57 readers.
    """

    def __init__(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(path)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        self._chunk_size = chunk_size
        self._reader = None

    @property
    def chunk_size(self) -> int:
        """
        Number of points yielded per iteration.
        """
        return self._chunk_size

    @abstractmethod
    def _open(self) -> None:
        """
        Open the E57 file.
        """

    @abstractmethod
    def _iterate_scans(
        self,
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """
        Iterate over every scan contained in the E57 file.

        Yields
        ------
        tuple
            (source_id, scan_arrays)
        """

    @abstractmethod
    def _create_chunk(
        self,
        arrays: dict[str, np.ndarray],
        source_id: int,
    ) -> Chunk:
        """
        Convert NumPy arrays into a Chunk.
        """

    def __iter__(self) -> Iterator[Chunk]:

        self._open()

        for source_id, scan in self._iterate_scans():
            size = len(next(iter(scan.values())))

            for start in range(
                0,
                size,
                self._chunk_size,
            ):
                end = start + self._chunk_size

                arrays = {key: value[start:end] for key, value in scan.items()}

                yield self._create_chunk(
                    arrays,
                    source_id,
                )

    def close(self) -> None:
        """
        Release resources.
        """
        self._reader = None
