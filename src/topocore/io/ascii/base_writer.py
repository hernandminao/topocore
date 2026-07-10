"""
topocore.io.ascii.base_writer
=============================

Base implementation for ASCII point cloud writers.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path

from topocore.io.base import PointCloudWriter


class BaseASCIIWriter(PointCloudWriter):
    """
    Base class for ASCII point cloud writers.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(path)

        self._encoding = encoding

    @property
    def encoding(self) -> str:
        """Output file encoding."""
        return self._encoding

    def close(self) -> None:
        """
        Release resources.

        ASCII writers do not keep persistent resources open.
        """
        pass
