"""
topocore.pointcloud.chunk
=========================

Chunk representation used by TopoCore.

A Chunk stores a fixed number of points and their attributes using
NumPy arrays. Large point clouds are represented as one or more chunks
to enable streaming and memory-efficient processing.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence

import numpy as np

from .attributes import ATTRIBUTE_DEFINITIONS
from .attributes import PointAttribute


class Chunk:
    """
    Stores a subset of a point cloud.

    Parameters
    ----------
    size
        Number of points contained in the chunk.

    attributes
        Attributes stored for every point.

    source_id
        Identifier of the data source that generated this chunk.

        Examples
        --------
        * LAS / LAZ -> 0
        * E57 -> scan index
        * Merged datasets -> source file identifier
    """

    def __init__(
        self,
        size: int,
        attributes: Sequence[PointAttribute],
        *,
        source_id: int = 0,
    ) -> None:
        """
        Create a new Chunk.
        """

        if size < 0:
            raise ValueError("size must be non-negative.")

        unique_attributes = tuple(dict.fromkeys(attributes))

        if len(unique_attributes) != len(attributes):
            raise ValueError("Duplicate attributes are not allowed.")

        self._size = size
        self._source_id = source_id
        self._attributes = unique_attributes

        self._data: dict[
            PointAttribute,
            np.ndarray,
        ] = {}

        for attribute in unique_attributes:
            info = ATTRIBUTE_DEFINITIONS[attribute]

            # A (1,) attribute (X, INTENSITY, CLASSIFICATION, ...) is
            # stored as a flat (size,) array. A multi-component
            # attribute such as COLOR or NORMAL, with shape (3,),
            # is stored as (size, 3) -- one row per point, not
            # squeezed into a single (size,) array that could only
            # ever hold one of its components.
            array_shape = (size,) if info.shape == (1,) else (size, *info.shape)

            self._data[attribute] = np.empty(
                array_shape,
                dtype=info.dtype,
            )

    @property
    def size(self) -> int:
        """
        Number of points stored in the chunk.
        """
        return self._size

    @property
    def source_id(self) -> int:
        """
        Identifier of the source that generated this chunk.
        """
        return self._source_id

    @property
    def attributes(
        self,
    ) -> tuple[PointAttribute, ...]:
        """
        Attributes available in the chunk.

        The returned tuple is immutable and cached during
        construction.
        """
        return self._attributes

    def has_attribute(
        self,
        attribute: PointAttribute,
    ) -> bool:
        """
        Return whether an attribute exists.
        """
        return attribute in self._data

    def __getitem__(
        self,
        attribute: PointAttribute,
    ) -> np.ndarray:
        """
        Return the NumPy array associated with an attribute.

        Raises
        ------
        KeyError
            If the attribute is not present.
        """
        return self._data[attribute]

    def __len__(self) -> int:
        """
        Return the number of points stored in the chunk.
        """
        return self._size

    def __iter__(
        self,
    ) -> Iterator[PointAttribute]:
        """
        Iterate over the available attributes.
        """
        return iter(self._attributes)

    def __contains__(
        self,
        attribute: PointAttribute,
    ) -> bool:
        """
        Return whether an attribute is stored in the chunk.
        """
        return attribute in self._data

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation.
        """

        attrs = ", ".join(attribute.name for attribute in self._attributes)

        return f"{self.__class__.__name__}(size={self._size}, source_id={self._source_id}, attributes=[{attrs}])"


__all__ = [
    "Chunk",
]
