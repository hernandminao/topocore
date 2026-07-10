"""
topocore.io.ascii.base_reader
=============================

Base implementation for ASCII point cloud readers.

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
from pathlib import Path

from topocore.io.base import PointCloudReader
from topocore.io.constants import DEFAULT_CHUNK_SIZE
from topocore.pointcloud.chunk import Chunk


class BaseASCIIReader(PointCloudReader):
    """
    Base implementation for ASCII readers.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        encoding: str = "utf-8",
    ) -> None:

        super().__init__(path)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if not encoding:
            raise ValueError("encoding cannot be empty.")

        self._chunk_size = chunk_size
        self._encoding = encoding

    @property
    def chunk_size(self) -> int:
        """
        Number of points per yielded chunk.
        """

        return self._chunk_size

    @property
    def encoding(self) -> str:
        """
        File encoding.
        """

        return self._encoding

    @abstractmethod
    def __iter__(self) -> Iterator[Chunk]:
        """
        Iterate over point cloud chunks.
        """

    def close(self) -> None:
        """
        Release reader resources.

        ASCII readers do not keep persistent resources open.
        """
