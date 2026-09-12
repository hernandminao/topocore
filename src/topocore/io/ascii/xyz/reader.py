"""
topocore.io.ascii.xyz.reader
============================

Reader for XYZ point cloud files.

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
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud


class XYZReader(BaseASCIIReader):
    """
    Reader for XYZ point cloud files.
    """

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

        self._parser = ASCIIParser(
            path,
            fmt=ASCIIFormat(
                delimiter=None,
                has_header=None,
                encoding=encoding,
            ),
        )

        self._converter = ASCIIConverter()

    def __iter__(self) -> Iterator[Chunk]:

        for batch in self._parser.iter_batches(
            chunk_size=self.chunk_size,
        ):
            yield self._converter(batch)

    def read(self) -> PointCloud:
        """
        Same base contract as `PointCloudReader.read()` -- no new
        parameter, no signature change. The one addition: if a
        `.prj` sidecar exists next to this file and resolves to a
        real CRS (see `topocore.io.crs.ExternalCRSDetector`), the
        resulting `PointCloud.crs` is set accordingly. XYZ has no
        internal CRS mechanism of its own (confirmed during this
        project's own audit -- plain X/Y/Z text carries no CRS
        concept at all), so `.prj` is the only source this format can
        ever get a CRS from. No sidecar, or one that doesn't resolve,
        leaves the result identical to the base behavior -- this can
        never turn a successful read into a failure.
        """
        return apply_external_crs(super().read(), self.path)

    @property
    def format_name(self) -> str:
        return "XYZ"
