"""
topocore.pointcloud.numpy_memory
================================

NumPy-based storage backend for TopoCore.

This module provides the default in-memory representation of a point
cloud using a Structure of Arrays (SoA) layout. Each point attribute
is stored as an independent NumPy array, allowing efficient vectorized
operations and cache-friendly access patterns.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.attributes import (
    ATTRIBUTE_DEFINITIONS,
    PointAttribute,
)


class NumpyMemory:
    """
    NumPy storage backend.

    Data are stored as a Structure of Arrays (SoA), where each
    PointAttribute owns an independent NumPy array.

    Examples
    --------
    >>> memory = NumpyMemory(
    ...     size=100,
    ...     attributes=[
    ...         PointAttribute.X,
    ...         PointAttribute.Y,
    ...         PointAttribute.Z,
    ...     ],
    ... )

    >>> memory[PointAttribute.X]
    array(...)
    """

    def __init__(
        self,
        size: int = 0,
        attributes: Iterable[PointAttribute] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        size
            Number of points.

        attributes
            Attributes that should be allocated.
        """

        if size < 0:
            raise ValueError(
                "size must be greater than or equal to zero."
            )

        self._size = size

        self._data: dict[
            PointAttribute,
            NDArray[Any],
        ] = {}

        if attributes is not None:
            for attribute in attributes:
                self.add_attribute(attribute)

    @property
    def size(self) -> int:
        """
        Number of stored points.
        """
        return self._size

    @property
    def attributes(
        self,
    ) -> frozenset[PointAttribute]:
        """
        Returns allocated attributes.
        """
        return frozenset(self._data.keys())

    def __len__(self) -> int:
        """
        Number of stored points.
        """
        return self._size

    def __contains__(
        self,
        attribute: object,
    ) -> bool:
        """
        Returns whether an attribute exists.
        """
        return (
            isinstance(attribute, PointAttribute)
            and attribute in self._data
        )

    def __getitem__(
        self,
        attribute: PointAttribute,
    ) -> NDArray[Any]:
        """
        Returns the NumPy array associated with an attribute.

        Raises
        ------
        KeyError
            If the attribute has not been allocated.
        """
        return self._data[attribute]

    def __setitem__(
        self,
        attribute: PointAttribute,
        values: NDArray[Any],
    ) -> None:
        """
        Replace the values of an existing attribute.

        Parameters
        ----------
        attribute
            Attribute to replace.

        values
            NumPy array containing the new values.

        Raises
        ------
        KeyError
            If the attribute has not been allocated.

        ValueError
            If the array shape is incompatible.

        TypeError
            If the array dtype is incompatible.
        """

        if attribute not in self._data:
            raise KeyError(
                f"Attribute '{attribute.name}' has not been allocated."
            )

        info = ATTRIBUTE_DEFINITIONS[attribute]

        expected_shape = (self._size, *info.shape)

        if values.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, "
                f"received {values.shape}."
            )

        expected_dtype = info.dtype

        if values.dtype != expected_dtype:
            raise TypeError(
                f"Expected dtype {expected_dtype}, "
                f"received {values.dtype}."
            )

        self._data[attribute] = values

    def add_attribute(
        self,
        attribute: PointAttribute,
    ) -> None:
        """
        Allocate a new attribute.

        If the attribute already exists, the call has no effect.
        """

        if attribute in self._data:
            return

        info = ATTRIBUTE_DEFINITIONS[attribute]

        shape = (self._size, *info.shape)

        self._data[attribute] = np.zeros(
            shape=shape,
            dtype=info.dtype,
        )

    def remove_attribute(
        self,
        attribute: PointAttribute,
    ) -> None:
        """
        Remove an attribute.

        Missing attributes are ignored.
        """

        self._data.pop(attribute, None)

    def resize(
        self,
        size: int,
    ) -> None:
        """
        Resize all allocated attributes.

        Existing values are preserved up to the minimum size.
        New elements are initialized to zero.

        Parameters
        ----------
        size
            New number of points.

        Raises
        ------
        ValueError
            If size is negative.
        """

        if size < 0:
            raise ValueError(
                "size must be greater than or equal to zero."
            )

        if size == self._size:
            return

        previous_size = self._size

        for attribute, array in self._data.items():

            info = ATTRIBUTE_DEFINITIONS[attribute]

            new_shape = (size, *info.shape)

            resized = np.zeros(
                shape=new_shape,
                dtype=info.dtype,
            )

            copy_size = min(previous_size, size)

            if copy_size > 0:
                resized[:copy_size] = array[:copy_size]

            self._data[attribute] = resized

        self._size = size

    def clone(self) -> "NumpyMemory":
        """
        Create a deep copy of this storage.
        """

        cloned = NumpyMemory()

        cloned._size = self._size

        cloned._data = {
            attribute: values.copy()
            for attribute, values in self._data.items()
        }

        return cloned

    def clear(self) -> None:
        """
        Remove every attribute and release all allocated memory.
        """

        self._data.clear()

        self._size = 0

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation.
        """

        attributes = ", ".join(
            attribute.name
            for attribute in sorted(
                self._data,
                key=lambda attribute: attribute.name,
            )
        )

        return (
            f"{self.__class__.__name__}("
            f"size={self._size}, "
            f"attributes=[{attributes}]"
            f")"
        )


__all__ = [
    "NumpyMemory",
]