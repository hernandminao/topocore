"""
topocore.io.ascii.records
=========================

Intermediate data structures for ASCII readers.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from dataclasses import dataclass, field

from numpy.typing import NDArray
import numpy as np


@dataclass(slots=True)
class ASCIIRecordBatch:
    """
    Batch of parsed ASCII records.

    Each entry contains a NumPy array with all values for one column.

    Examples
    --------
    >>> batch["x"]
    >>> batch["classification"]
    >>> "red" in batch
    """

    columns: dict[str, NDArray[np.generic]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """
        Number of records stored in the batch.
        """
        if not self.columns:
            return 0

        return len(next(iter(self.columns.values())))

    def __len__(self) -> int:
        return self.size

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def __getitem__(self, name: str) -> NDArray[np.generic]:
        return self.columns[name]

    def get(
        self,
        name: str,
        default: NDArray[np.generic] | None = None,
    ) -> NDArray[np.generic] | None:
        return self.columns.get(name, default)

    def keys(self) -> KeysView[str]:
        return self.columns.keys()

    def values(self) -> ValuesView[NDArray[np.generic]]:
        return self.columns.values()

    def items(self) -> ItemsView[str, NDArray[np.generic]]:
        return self.columns.items()

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)
