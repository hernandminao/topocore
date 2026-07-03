"""
topocore.io.las.base_reader
===========================

Base implementation shared by LAS and LAZ readers.

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

from topocore.io.base import PointCloudReader
from topocore.pointcloud.chunk import Chunk


class BaseLASReader(PointCloudReader):
    """
    Base class for LAS-based readers.

    This class contains the common logic shared by LAS and LAZ readers.
    Concrete subclasses are responsible only for opening the underlying
    file.
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        super().__init__(path)

        self._reader = None
        self._header = None

    @abstractmethod
    def _open(self) -> None:
        """
        Open the underlying file.

        Implemented by subclasses.
        """

    @property
    def header(self):
        """
        Return the LAS header.
        """

        return self._header

    def __iter__(self) -> Iterator[Chunk]:
        """
        Iterate over chunks contained in the file.
        """

        self._open()

        yield from self._iterate_chunks()

    @abstractmethod
    def _iterate_chunks(self) -> Iterator[Chunk]:
        """
        Yield point cloud chunks.
        """

    def close(self) -> None:
        """
        Close the underlying reader.
        """

        if self._reader is not None:
            self._reader.close()
            self._reader = None