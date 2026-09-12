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
from topocore.io.crs.external import apply_external_crs
from topocore.io.exceptions import CorruptedFileError
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud


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
            # PTS requires the first non-empty line to contain the
            # declared point count.
            #

            first_line = stream.readline()
            if not first_line:
                raise CorruptedFileError("PTS file is missing its point count.")

            try:
                declared_count = int(first_line.strip())
            except ValueError as exc:
                raise CorruptedFileError("PTS point count must be an integer.") from exc

            if declared_count < 0:
                raise CorruptedFileError("PTS point count cannot be negative.")

            parser = ASCIIParser(
                stream,
                fmt=ASCIIFormat(
                    delimiter=None,
                    has_header=False,
                    encoding=self.encoding,
                ),
            )

            converter = ASCIIConverter()

            actual_count = 0

            for batch in parser.iter_batches(
                chunk_size=self.chunk_size,
            ):
                actual_count += batch.size
                yield converter(batch)

            if actual_count != declared_count:
                raise CorruptedFileError(
                    f"PTS point count mismatch: header declares {declared_count}, read {actual_count} points."
                )

    def read(self) -> PointCloud:
        """
        Same base contract as `PointCloudReader.read()` -- no new
        parameter, no signature change. The one addition: if a
        `.prj` sidecar exists next to this file and resolves to a
        real CRS (see `topocore.io.crs.ExternalCRSDetector`), the
        resulting `PointCloud.crs` is set accordingly. PTS has no
        internal CRS mechanism of its own (confirmed during this
        project's own audit -- a leading point count plus plain
        X/Y/Z text carries no CRS concept at all), so `.prj` is the
        only source this format can ever get a CRS from. No sidecar,
        or one that doesn't resolve, leaves the result identical to
        the base behavior -- this can never turn a successful read
        into a failure.
        """
        return apply_external_crs(super().read(), self.path)

    @property
    def format_name(self) -> str:
        return "PTS"
