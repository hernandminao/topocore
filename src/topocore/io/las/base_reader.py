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
from pathlib import Path
from typing import Any

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
        path: str | Path,
    ) -> None:
        super().__init__(path)

        # laspy ships no type stubs / py.typed marker: ``Any`` is the
        # accurate type for its reader and header objects here, not
        # a placeholder for "didn't bother typing this."
        self._reader: Any = None
        self._header: Any = None

    @abstractmethod
    def _open(self) -> None:
        """
        Open the underlying file.

        Implemented by subclasses.
        """

    @property
    def header(self) -> Any:
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


__all__ = [
    "BaseLASReader",
]
