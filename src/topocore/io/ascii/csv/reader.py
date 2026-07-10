"""
topocore.io.ascii.csv.reader
============================

Reader for CSV point cloud files.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from topocore.io.ascii.base_reader import BaseASCIIReader
from topocore.io.ascii.converter import ASCIIConverter
from topocore.io.ascii.format import ASCIIFormat
from topocore.io.ascii.parser import ASCIIParser
from topocore.pointcloud.chunk import Chunk


class CSVReader(BaseASCIIReader):
    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int = 1_000_000,
        encoding: str = "utf-8",
        has_header: bool | None = None,
    ) -> None:

        super().__init__(
            path,
            chunk_size=chunk_size,
            encoding=encoding,
        )

        self._parser = ASCIIParser(
            path,
            fmt=ASCIIFormat(
                delimiter=",",
                has_header=has_header,
                encoding=encoding,
            ),
        )

        self._converter = ASCIIConverter()

    def __iter__(self) -> Iterator[Chunk]:

        for batch in self._parser.iter_batches(
            chunk_size=self.chunk_size,
        ):
            yield self._converter(batch)

    @property
    def format_name(self) -> str:
        return "CSV"
