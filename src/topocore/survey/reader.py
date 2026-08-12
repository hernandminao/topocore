"""
topocore.survey.reader
=========================

Reader for total-station / GNSS survey text files.

Produces a ``SurveyPointSet``, not a ``topocore.pointcloud.Chunk``
(see ``topocore.survey.models`` for why). This is a small,
self-contained parser rather than a reuse of
``topocore.io.ascii.ASCIIParser``: that parser has no column slot for
a point ID or a field code in the headerless case (its
``ASCIIFormat`` only carries ``x_column``/``y_column``/``z_column``),
Core/IO is frozen for this project (see the project roadmap), and its
header-alias table (``topocore.io.ascii.header.HEADER_ALIASES``) only
recognizes English column names -- "Punto,Este,Norte,Z,Descripcion",
the standard header on most Latin American total-station exports and
this project's own primary use case, would not be detected. Header
recognition here is bilingual (ES/EN) instead.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Final

from topocore.survey.exceptions import SurveyFormatError, SurveyRecordError
from topocore.survey.formats import ColumnLayout, SurveyFormat, column_layout
from topocore.survey.models import SurveyPoint, SurveyPointSet

_COMMENT_PREFIXES = ("#", "//")
_CANDIDATE_DELIMITERS = (",", ";", "\t")

#: Bilingual (ES/EN) header aliases, normalized (lowercase, no
#: spaces/underscores/dashes) before matching.
_HEADER_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "id": ("id", "punto", "pto", "point", "pointid", "station"),
    "x": ("x", "este", "east", "easting"),
    "y": ("y", "norte", "north", "northing"),
    "z": ("z", "cota", "elevacion", "elevation", "elev", "level", "rl"),
    "code": (
        "code",
        "codigo",
        "descripcion",
        "description",
        "desc",
        "obs",
        "observacion",
        "feature",
    ),
}

#: A numbered raw line: (1-based physical line number in the source
#: file, stripped text) -- kept separate from any per-data-row
#: sequential fallback ID, since after removing comments/blank lines
#: the two no longer coincide.
_NumberedLine = tuple[int, str]
_NumberedRow = tuple[int, list[str]]


def _normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _detect_header_columns(fields: list[str]) -> dict[str, int]:
    columns: dict[str, int] = {}

    for index, value in enumerate(fields):
        normalized = _normalize(value)

        for name, aliases in _HEADER_ALIASES.items():
            if normalized in aliases:
                columns[name] = index
                break

    return columns


def _detect_delimiter(line: str) -> str | None:
    for delimiter in _CANDIDATE_DELIMITERS:
        if delimiter in line:
            return delimiter

    return None


def _split(line: str, delimiter: str | None) -> list[str]:
    """
    Split one data line into fields.

    For comma/semicolon/tab, uses `csv.reader` (quote-aware) rather
    than a plain `str.split()`, so a quoted field containing the
    delimiter itself (e.g. a description like
    ``"CERCA, COSTADO NORTE"`` in a comma-delimited file) is not
    incorrectly split into extra fields.
    """
    if delimiter is None:
        return line.split()

    reader = csv.reader(io.StringIO(line), delimiter=delimiter)
    fields = next(reader)
    return [field.strip() for field in fields]


def _read_data_lines(path: Path, encoding: str) -> list[_NumberedLine]:
    """
    Read non-blank, non-comment lines, each tagged with its 1-based
    physical line number in the source file (so error messages and
    future diagnostics can point back to the real file, even after
    comments/blank lines have been filtered out).
    """
    with path.open("r", encoding=encoding) as stream:
        raw_lines = stream.readlines()

    lines: list[_NumberedLine] = []

    for source_line, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(_COMMENT_PREFIXES):
            continue

        lines.append((source_line, line))

    return lines


def _layout_from_header(fields: list[str]) -> ColumnLayout | None:
    columns = _detect_header_columns(fields)

    if not {"x", "y", "z"}.issubset(columns):
        return None

    return ColumnLayout(
        x_column=columns["x"],
        y_column=columns["y"],
        z_column=columns["z"],
        id_column=columns.get("id"),
        code_column=columns.get("code"),
    )


def _build_point(
    row: list[str],
    layout: ColumnLayout,
    *,
    source_line: int,
    fallback_id: int,
) -> SurveyPoint:
    """
    Parameters
    ----------
    source_line
        1-based physical line number in the original file --
        used only for error messages, so they point to a line the
        user can actually go find.
    fallback_id
        1-based sequential position among data rows -- used only as
        the generated `SurveyPoint.id` when the file has no ID
        column (or the ID cell for this row is empty). Deliberately
        NOT the same value as `source_line`: an auto-generated ID is
        meant to just be a stable, unique label, not a claim about
        which physical line produced it.
    """
    try:
        x = float(row[layout.x_column])
        y = float(row[layout.y_column])
        z = float(row[layout.z_column])
    except (IndexError, ValueError) as exc:
        raise SurveyRecordError(f"Invalid coordinate values at line {source_line} of the survey file: {row}") from exc

    point_id = str(fallback_id)
    if layout.id_column is not None and layout.id_column < len(row):
        raw_id = row[layout.id_column].strip()
        if raw_id:
            point_id = raw_id
        # else: ID column exists but this row's cell is empty --
        # keep the sequential fallback rather than producing
        # SurveyPoint(id="", ...).

    code = None
    if layout.code_column is not None and layout.code_column < len(row):
        raw_code = row[layout.code_column].strip()
        code = raw_code or None

    return SurveyPoint(id=point_id, x=x, y=y, z=z, code=code)


class SurveyTXTReader:
    """
    Reads a total-station / GNSS survey text file into a
    ``SurveyPointSet``.

    Supports any delimited text layout: comma, semicolon, tab, or
    whitespace separated, with or without a header row. Column order
    is resolved, in this priority:

    1. An explicit ``format`` preset (see ``SurveyFormat``).
    2. Auto-detection from a recognized header row.

    If neither applies, raises ``SurveyFormatError`` rather than
    guessing.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        format: SurveyFormat | None = None,
        delimiter: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self._path = Path(path)
        self._format = format
        self._delimiter = delimiter
        self._encoding = encoding

    @property
    def path(self) -> Path:
        return self._path

    @property
    def format(self) -> SurveyFormat | None:
        return self._format

    def read(self) -> SurveyPointSet:
        """
        Read the file and return a ``SurveyPointSet``.

        Raises
        ------
        SurveyFormatError
            If no ``format`` preset was given and no header could be
            detected.
        SurveyRecordError
            If a data row has non-numeric or missing coordinates.
        """
        numbered_lines = _read_data_lines(self._path, self._encoding)

        if not numbered_lines:
            return SurveyPointSet(points=())

        delimiter = self._delimiter or _detect_delimiter(numbered_lines[0][1])
        numbered_rows: list[_NumberedRow] = [
            (source_line, _split(text, delimiter)) for source_line, text in numbered_lines
        ]

        layout, data_rows = self._resolve_layout(numbered_rows)

        points = tuple(
            _build_point(row, layout, source_line=source_line, fallback_id=index + 1)
            for index, (source_line, row) in enumerate(data_rows)
        )

        return SurveyPointSet(points=points)

    def _resolve_layout(
        self,
        numbered_rows: list[_NumberedRow],
    ) -> tuple[ColumnLayout, list[_NumberedRow]]:
        if self._format is not None:
            return column_layout(self._format), numbered_rows

        layout = _layout_from_header(numbered_rows[0][1])

        if layout is not None:
            return layout, numbered_rows[1:]

        raise SurveyFormatError(
            f"Could not detect a header in '{self._path.name}' and no "
            "format preset was given. Pass format=SurveyFormat.<...> "
            "explicitly."
        )


__all__ = [
    "SurveyTXTReader",
]
