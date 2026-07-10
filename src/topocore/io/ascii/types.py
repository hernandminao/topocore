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
from typing import TextIO

type ASCIIInput = str | Path | TextIO | Iterable[str]
