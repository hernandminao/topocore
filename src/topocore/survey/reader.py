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

from pathlib import Path
from typing import Final

from topocore.survey.exceptions import SurveyFormatError
from topocore.survey.exceptions import SurveyRecordError
from topocore.survey.formats import ColumnLayout
from topocore.survey.formats import SurveyFormat
from topocore.survey.formats import column_layout
from topocore.survey.models import SurveyPoint
from topocore.survey.models import SurveyPointSet

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
        "code", "codigo", "descripcion", "description", "desc", "obs",
        "observacion", "feature",
    ),
}


def _normalize(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


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
    if delimiter is None:
        return line.split()

    return [field.strip() for field in line.split(delimiter)]


def _read_data_lines(path: Path, encoding: str) -> list[str]:
    with path.open("r", encoding=encoding) as stream:
        raw_lines = stream.readlines()

    lines: list[str] = []

    for raw_line in raw_lines:
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(_COMMENT_PREFIXES):
            continue

        lines.append(line)

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
    fallback_id: int,
) -> SurveyPoint:
    try:
        x = float(row[layout.x_column])
        y = float(row[layout.y_column])
        z = float(row[layout.z_column])
    except (IndexError, ValueError) as exc:
        raise SurveyRecordError(
            f"Invalid coordinate values in row {fallback_id}: {row}"
        ) from exc

    point_id = str(fallback_id)
    if layout.id_column is not None and layout.id_column < len(row):
        point_id = row[layout.id_column]

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
        lines = _read_data_lines(self._path, self._encoding)

        if not lines:
            return SurveyPointSet(points=())

        delimiter = self._delimiter or _detect_delimiter(lines[0])
        rows = [_split(line, delimiter) for line in lines]

        layout, data_rows = self._resolve_layout(rows)

        points = tuple(
            _build_point(row, layout, index + 1)
            for index, row in enumerate(data_rows)
        )

        return SurveyPointSet(points=points)

    def _resolve_layout(
        self,
        rows: list[list[str]],
    ) -> tuple[ColumnLayout, list[list[str]]]:
        if self._format is not None:
            return column_layout(self._format), rows

        layout = _layout_from_header(rows[0])

        if layout is not None:
            return layout, rows[1:]

        raise SurveyFormatError(
            f"Could not detect a header in '{self._path.name}' and no "
            "format preset was given. Pass format=SurveyFormat.<...> "
            "explicitly."
        )


__all__ = [
    "SurveyTXTReader",
]
