"""
topocore.io.ascii.pts.reader
============================

Reader for PTS point cloud files.

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


class PTSReader(BaseASCIIReader):
    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int = 1_000_000,
        encoding: str = "utf-8",
    ) -> None:

        super().__init__(
            path,
            chunk_size=chunk_size,
            encoding=encoding,
        )

        self._path = Path(path)

    def __iter__(self) -> Iterator[Chunk]:

        with self._path.open(
            "r",
            encoding=self.encoding,
        ) as stream:
            #
            # Skip point count
            #

            next(stream)

            parser = ASCIIParser(
                stream,
                fmt=ASCIIFormat(
                    delimiter=None,
                    has_header=False,
                    encoding=self.encoding,
                ),
            )

            converter = ASCIIConverter()

            for batch in parser.iter_batches(
                chunk_size=self.chunk_size,
            ):
                yield converter(batch)

    @property
    def format_name(self) -> str:
        return "PTS"
