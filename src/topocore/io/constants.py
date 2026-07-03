"""
topocore.io.constants
=====================

Shared constants used by the TopoCore I/O subsystem.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final

#: Default number of points read per chunk when streaming point clouds.
DEFAULT_CHUNK_SIZE: Final[int] = 1_000_000

__all__ = [
    "DEFAULT_CHUNK_SIZE",
]