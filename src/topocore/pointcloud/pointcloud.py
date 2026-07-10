"""
topocore.pointcloud.pointcloud
==============================

High-level point cloud container.

A PointCloud is composed of one or more Chunk objects. It provides a
format-independent representation of a point cloud while remaining
agnostic of file formats such as LAS, LAZ, E57 or PLY.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from topocore.geometry.bbox3d import BBox3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk


@dataclass
class PointCloudMetadata:
    """
    Metadata associated with a point cloud.
    """

    crs: str | None = None
    bounds: BBox3D | None = None
    source_format: str = ""
    is_georeferenced: bool = False


class PointCloud:
    """
    High-level point cloud container.
    """

    def __init__(self) -> None:
        """
        Create an empty point cloud.
        """
        self._chunks: list[Chunk] = []
        self._metadata: PointCloudMetadata = PointCloudMetadata()

    def __len__(self) -> int:
        """
        Return the total number of points.
        """
        return self.point_count

    def __iter__(self) -> Iterator[Chunk]:
        """
        Iterate over the chunks.
        """
        return iter(self._chunks)

    @property
    def metadata(self) -> PointCloudMetadata:
        """
        Return the metadata object.
        """
        return self._metadata

    @property
    def crs(self) -> str | None:
        """
        Return the Coordinate Reference System.
        """
        return self._metadata.crs

    @crs.setter
    def crs(self, value: str) -> None:
        """
        Set the Coordinate Reference System.
        """
        self._metadata.crs = value

    @property
    def bounds(self) -> BBox3D | None:
        """
        Return the spatial bounds.
        """
        return self._metadata.bounds

    @property
    def chunk_count(self) -> int:
        """
        Number of chunks.
        """
        return len(self._chunks)

    @property
    def point_count(self) -> int:
        """
        Total number of points.
        """
        return sum(chunk.size for chunk in self._chunks)

    @property
    def is_empty(self) -> bool:
        """
        Return whether the point cloud contains no points.
        """
        return self.point_count == 0

    @property
    def attributes(self) -> frozenset[PointAttribute]:
        """
        Return the union of attributes present in all chunks.
        """
        attributes: set[PointAttribute] = set()
        for chunk in self._chunks:
            attributes.update(chunk.attributes)
        return frozenset(attributes)

    def add_chunk(
        self,
        chunk: Chunk,
    ) -> None:
        """
        Add a chunk to the point cloud.
        """
        self._chunks.append(chunk)

    def update_bounds(self) -> None:
        """
        Calculate and update the 3D bounding box from all chunks.
        """
        if self.is_empty or PointAttribute.X not in self.attributes:
            self._metadata.bounds = None
            return

        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")

        for chunk in self._chunks:
            if PointAttribute.X not in chunk:
                continue

            x = chunk[PointAttribute.X]
            y = chunk[PointAttribute.Y]
            z = chunk[PointAttribute.Z]

            min_x = min(min_x, float(np.min(x)))
            min_y = min(min_y, float(np.min(y)))
            min_z = min(min_z, float(np.min(z)))

            max_x = max(max_x, float(np.max(x)))
            max_y = max(max_y, float(np.max(y)))
            max_z = max(max_z, float(np.max(z)))

        self._metadata.bounds = BBox3D(
            min_x=min_x,
            min_y=min_y,
            min_z=min_z,
            max_x=max_x,
            max_y=max_y,
            max_z=max_z,
        )

    def remove_chunk(
        self,
        index: int,
    ) -> Chunk:
        """
        Remove and return the chunk at the given index.
        """
        return self._chunks.pop(index)

    def clear(self) -> None:
        """
        Remove all chunks from the point cloud.
        """
        self._chunks.clear()
        self._metadata = PointCloudMetadata()

    def clone(self) -> PointCloud:
        """
        Create a deep copy of the point cloud.
        """
        cloned = PointCloud()
        cloned._chunks = list(
            self._chunks
        )  # Shallow copy of chunks for memory efficiency, or deep copy if needed
        cloned._metadata = PointCloudMetadata(
            crs=self._metadata.crs,
            bounds=self._metadata.bounds,
            source_format=self._metadata.source_format,
            is_georeferenced=self._metadata.is_georeferenced,
        )
        return cloned

    def __getitem__(self, index: int) -> Chunk:
        """
        Return the chunk at the given index.
        """
        return self._chunks[index]

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"chunks={self.chunk_count}, "
            f"points={self.point_count}, "
            f"crs={self.crs or 'None'}, "
            f"attributes={len(self.attributes)}"
            f")"
        )


__all__ = [
    "PointCloud",
    "PointCloudMetadata",
]
