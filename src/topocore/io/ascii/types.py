"""
topocore.io.ascii.types
=======================

Common type aliases used by ASCII readers.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol


class TextReadable(Protocol):
    """
    Protocol for readable text streams.
    """

    def read(
        self,
        size: int = -1,
    ) -> str: ...

    def readline(
        self,
        size: int = -1,
    ) -> str: ...

    def __iter__(self) -> Iterable[str]: ...


type ASCIIInput = str | Path | TextReadable | Iterable[str]
