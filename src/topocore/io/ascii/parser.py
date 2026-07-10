"""
topocore.io.ascii.parser
========================

Generic parser for ASCII-based point cloud formats.

Supported formats
-----------------
- XYZ
- CSV
- TXT
- ASC
- PTS

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .exceptions import InvalidASCIIRecordError
from .format import ASCIIFormat
from .header import HeaderDetector
from .records import ASCIIRecordBatch
from .type_inference import TypeInferer
from .types import ASCIIInput


@dataclass(slots=True)
class ParserContext:
    """
    Internal parser state.
    """

    delimiter: str | None

    columns: dict[str, int]

    buffers: dict[str, list[str]]

    @property
    def size(self) -> int:

        if not self.buffers:
            return 0

        return len(next(iter(self.buffers.values())))

    def clear(self) -> None:

        for values in self.buffers.values():
            values.clear()


class ASCIIParser:
    """
    Generic parser for ASCII point cloud files.
    """

    def __init__(
        self,
        source: ASCIIInput,
        *,
        fmt: ASCIIFormat,
    ) -> None:

        self._source = source

        self._format = fmt

    @contextmanager
    def _open_source(
        self,
    ) -> Iterator[TextIO]:
        """
        Open the configured source.

        Supports file paths and already opened text streams.
        """

        if isinstance(self._source, Path):
            with self._source.open(
                "r",
                encoding=self._format.encoding,
            ) as stream:
                yield stream

            return

        if isinstance(self._source, str):
            with open(
                self._source,
                encoding=self._format.encoding,
            ) as stream:
                yield stream

            return

        #
        # Existing text stream.
        #

        if hasattr(self._source, "read"):
            yield self._source

            return

        #
        # Iterable[str]
        #

        from io import StringIO

        yield StringIO(
            "".join(self._source),
        )

    def _iter_data_lines(
        self,
        stream: TextIO,
    ) -> Iterator[str]:

        for raw_line in stream:
            line = raw_line.strip()

            if not line:
                continue

            if any(line.startswith(prefix) for prefix in self._format.comment_prefixes):
                continue

            yield line

    @staticmethod
    def _detect_delimiter(
        line: str,
    ) -> str | None:

        for delimiter in (
            ",",
            ";",
            "\t",
        ):
            if delimiter in line:
                return delimiter

        #
        # Whitespace separated.
        #

        return None

    @staticmethod
    def _split(
        line: str,
        delimiter: str | None,
    ) -> list[str]:

        if delimiter is None:
            return line.split()

        return [value.strip() for value in line.split(delimiter)]

    def _default_columns(
        self,
    ) -> dict[str, int]:

        return {
            "x": self._format.x_column,
            "y": self._format.y_column,
            "z": self._format.z_column,
        }

    def _initialize_context(
        self,
        first_line: str,
    ) -> ParserContext:

        delimiter = self._format.delimiter or self._detect_delimiter(first_line)

        fields = self._split(
            first_line,
            delimiter,
        )

        detection = HeaderDetector.detect(
            fields,
        )

        has_header = self._format.has_header is not False and detection.has_header

        columns = detection.columns if has_header else self._default_columns()

        context = ParserContext(
            delimiter=delimiter,
            columns=columns,
            buffers={name: [] for name in columns},
        )

        if not has_header:
            self._append_record(
                context,
                fields,
                first_line,
            )

        return context

    def iter_batches(
        self,
        *,
        chunk_size: int,
    ) -> Iterator[ASCIIRecordBatch]:
        """
        Iterate over the source yielding parsed batches.

        Parameters
        ----------
        chunk_size
            Maximum number of records per batch.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        with self._open_source() as stream:
            data_lines = self._iter_data_lines(stream)

            try:
                first_line = next(data_lines)

            except StopIteration:
                return

            context = self._initialize_context(
                first_line,
            )

            for line in data_lines:
                fields = self._split(
                    line,
                    context.delimiter,
                )

                self._append_record(
                    context,
                    fields,
                    line,
                )

                if context.size >= chunk_size:
                    yield self._emit_batch(
                        context,
                    )

            if context.size:
                yield self._emit_batch(
                    context,
                )

    def _append_record(
        self,
        context: ParserContext,
        fields: list[str],
        line: str,
    ) -> None:
        """
        Append one record to the current buffers.

        Missing trailing fields are stored as empty strings.
        """

        buffers = context.buffers

        try:
            for (
                column_name,
                column_index,
            ) in context.columns.items():
                value = fields[column_index] if column_index < len(fields) else ""

                buffers[column_name].append(
                    value,
                )

        except (IndexError, ValueError) as exc:
            raise InvalidASCIIRecordError(f"Invalid ASCII record: {line}") from exc

    def _emit_batch(
        self,
        context: ParserContext,
    ) -> ASCIIRecordBatch:
        """
        Emit the current batch and clear internal buffers.
        """

        batch = self._build_batch(
            context.buffers,
        )

        context.clear()

        return batch

    @staticmethod
    def _build_batch(
        buffers: dict[str, list[str]],
    ) -> ASCIIRecordBatch:
        """
        Build a typed ASCIIRecordBatch.
        """

        columns = {
            name: TypeInferer.infer(
                name,
                values,
            )
            for (
                name,
                values,
            ) in buffers.items()
        }

        return ASCIIRecordBatch(
            columns=columns,
        )
