"""
topocore.io.ply.reader
======================

Reader for PLY point cloud files.

Supports:

- ASCII
- Binary Little Endian
- Binary Big Endian

using lazy chunked iteration.

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

import numpy as np

from topocore.io.base import PointCloudReader
from topocore.io.common.records import PointRecordBatch
from topocore.pointcloud.chunk import Chunk

from .converter import PLYConverter
from .enums import PLYFormat
from .exceptions import InvalidPLYError
from .header import PLYElement
from .header_parser import PLYHeaderParser


class PLYReader(PointCloudReader):
    """
    Reads PLY files.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int = 1_000_000,
    ) -> None:

        super().__init__(path)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        self._chunk_size = chunk_size

        self._stream = None

        self._header = None

        self._converter = PLYConverter()

    def _ensure_open(self) -> None:
        if self._stream is None or self._header is None:
            self._open()

    def __iter__(self) -> Iterator[Chunk]:
        """
        Iterate over the point cloud chunks.
        """
        self._ensure_open()

        try:
            vertex = self._vertex_element()

            if self._header.format is PLYFormat.ASCII:
                yield from self._iter_ascii(vertex)

            else:
                yield from self._iter_binary(vertex)

        finally:
            self.close()

    def close(self):

        if self._stream is not None:
            self._stream.close()

            self._stream = None

    def _open(self):

        if self._stream is not None and self._header is not None:
            return

        self._stream = open(  # noqa: SIM115
            self.path, "rb"
        )

        self._header = PLYHeaderParser.parse(self._stream)

    def _vertex_element(self) -> PLYElement:

        if self._header is None:
            raise InvalidPLYError("PLY header not loaded. Did you call _open()?")

        for element in self._header.elements:
            if element.name == "vertex":
                return element

        raise InvalidPLYError("PLY file contains no vertex element.")

    def _iter_ascii(
        self,
        vertex: PLYElement,
    ):

        properties = [
            p
            for p in vertex.properties
            if hasattr(
                p,
                "dtype",
            )
        ]

        arrays = {p.name: [] for p in properties}

        points = 0

        total = vertex.count

        while points < total:
            raw = self._stream.readline()

            if raw == b"":
                break

            try:
                values = raw.decode("utf-8").split()

            except UnicodeDecodeError as exc:
                raise InvalidPLYError("PLY header is not valid UTF-8.") from exc

            if not values:
                continue

            for index, prop in enumerate(properties):
                arrays[prop.name].append(values[index])

            points += 1

            if points % self._chunk_size == 0:
                yield self._flush_ascii(
                    arrays,
                )

                arrays = {p.name: [] for p in properties}

        if arrays[properties[0].name]:
            yield self._flush_ascii(
                arrays,
            )

    def _flush_ascii(
        self,
        arrays,
    ):

        converted = {}

        vertex = self._vertex_element()

        for prop in vertex.properties:
            if not hasattr(
                prop,
                "dtype",
            ):
                continue

            converted[prop.name] = np.asarray(
                arrays[prop.name],
                dtype=prop.dtype.numpy_dtype,
            )

        batch = PointRecordBatch(
            arrays=converted,
        )

        return self._converter.convert(
            batch,
        )

    def _iter_binary(
        self,
        vertex: PLYElement,
    ):

        properties = [
            p
            for p in vertex.properties
            if hasattr(
                p,
                "dtype",
            )
        ]

        endian = "<" if self._header.format is PLYFormat.BINARY_LITTLE_ENDIAN else ">"

        dtype = np.dtype(
            [
                (
                    p.name,
                    endian + p.dtype.numpy_dtype.str[1:],
                )
                for p in properties
            ]
        )

        remaining = vertex.count

        while remaining > 0:
            size = min(
                self._chunk_size,
                remaining,
            )

            data = np.fromfile(
                self._stream,
                dtype=dtype,
                count=size,
            )

            if len(data) == 0:
                break

            arrays = {name: data[name] for name in data.dtype.names}

            batch = PointRecordBatch(
                arrays=arrays,
            )

            yield self._converter.convert(
                batch,
            )

            remaining -= len(data)
