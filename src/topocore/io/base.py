"""
topocore.io.base
================

Abstract base classes for point cloud readers and writers.

These classes define the common API implemented by every supported
point cloud format.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud


class PointCloudReader(ABC):
    """
    Abstract base class for point cloud readers.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        """
        Initialize the reader.

        Parameters
        ----------
        path
            Path to the input file.
        """

        self._path = Path(path)

    @property
    def path(self) -> Path:
        """
        Return the input file path.
        """

        return self._path

    @abstractmethod
    def __iter__(self) -> Iterator[Chunk]:
        """
        Iterate over chunks contained in the file.
        """

    def read(self) -> PointCloud:
        """
        Read the complete file into memory.

        Returns
        -------
        PointCloud
            Loaded point cloud.
        """

        cloud = PointCloud()

        for chunk in self:
            cloud.add_chunk(chunk)

        return cloud

    @abstractmethod
    def close(self) -> None:
        """
        Release resources associated with the reader.
        """

    def __enter__(self) -> PointCloudReader:
        """
        Enter the runtime context.
        """

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        Exit the runtime context.
        """

        self.close()


class PointCloudWriter(ABC):
    """
    Abstract base class for point cloud writers.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        """
        Initialize the writer.

        Parameters
        ----------
        path
            Destination file.
        """

        self._path = Path(path)

    @property
    def path(self) -> Path:
        """
        Return the destination path.
        """

        return self._path

    @abstractmethod
    def write(
        self,
        cloud: PointCloud,
    ) -> None:
        """
        Write a point cloud.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Release writer resources.
        """

    def __enter__(self) -> PointCloudWriter:
        """
        Enter the runtime context.
        """

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        Exit the runtime context.
        """

        self.close()


__all__ = [
    "PointCloudReader",
    "PointCloudWriter",
]
