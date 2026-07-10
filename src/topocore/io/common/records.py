"""
topocore.io.common.records
==========================

Common record batch representation used by point cloud readers.

A PointRecordBatch represents a homogeneous batch of point attributes
read from a source file before conversion into a Chunk.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class PointRecordBatch:
    """
    Batch of point records.

    Parameters
    ----------
    arrays
        Mapping from attribute/property name to NumPy array.

    source_id
        Source identifier.

    Notes
    -----
    All arrays must have identical length.
    """

    arrays: dict[str, np.ndarray]

    source_id: int = 0

    _size: int = field(init=False, repr=False)

    _attributes: frozenset[str] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:

        if not self.arrays:
            self._size = 0

            self._attributes = frozenset()

            return

        iterator = iter(self.arrays.values())

        first = next(iterator)

        size = len(first)

        for array in iterator:
            if len(array) != size:
                raise ValueError("All arrays must have identical length.")

        self._size = size

        self._attributes = frozenset(self.arrays.keys())

    @property
    def size(
        self,
    ) -> int:
        """
        Number of records.
        """
        return self._size

    @property
    def attributes(
        self,
    ) -> frozenset[str]:
        """
        Available attribute names.
        """
        return self._attributes

    @property
    def is_empty(
        self,
    ) -> bool:
        """
        Return whether the batch contains records.
        """
        return self._size == 0

    def has(
        self,
        name: str,
    ) -> bool:
        """
        Return whether an attribute exists.
        """
        return name in self.arrays

    def get(
        self,
        name: str,
    ) -> np.ndarray | None:
        """
        Return an attribute array.
        """
        return self.arrays.get(name)

    def keys(
        self,
    ) -> tuple[str, ...]:
        """
        Return attribute names.
        """
        return tuple(self.arrays.keys())

    def values(
        self,
    ) -> tuple[np.ndarray, ...]:
        """
        Return attribute arrays.
        """
        return tuple(self.arrays.values())

    def items(
        self,
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """
        Return (name, array) pairs.
        """
        return tuple(self.arrays.items())

    def __contains__(
        self,
        name: str,
    ) -> bool:

        return name in self.arrays

    def __getitem__(
        self,
        name: str,
    ) -> np.ndarray:

        return self.arrays[name]

    def __iter__(
        self,
    ) -> Iterator[str]:

        return iter(self.arrays)

    def __len__(
        self,
    ) -> int:

        return self._size

    def __repr__(
        self,
    ) -> str:

        attrs = ", ".join(sorted(self.arrays))

        return (
            f"{self.__class__.__name__}("
            f"size={self._size}, "
            f"source_id={self.source_id}, "
            f"attributes=[{attrs}]"
            f")"
        )


__all__ = [
    "PointRecordBatch",
]
