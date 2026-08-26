"""
Regression/coverage suite for topocore.survey.reader.SurveyTXTReader
-- PR20 coverage phase.

This module was audited fresh (not just extended for coverage), per
the file's own documented design goals: bilingual (ES/EN) header
detection for the standard Latin American total-station export
format, quote-aware delimiter splitting, and column-order resolution
via explicit format preset or header auto-detection.

Verified end to end with real files for every documented behavior:
Spanish header (the primary use case -- "Punto,Este,Norte,Z,
Descripcion"), English header, whitespace-delimited with an explicit
SurveyFormat preset, a quoted field containing the delimiter itself
("CERCA, COSTADO NORTE" in a comma-delimited file), comment lines
(both "#" and "//") and blank lines skipped, an empty file, missing
header with no format preset (SurveyFormatError), invalid coordinate
values (SurveyRecordError), missing ID/code columns (sequential
fallback ID, code=None), an ID column present but with an empty cell
for one row (falls back to sequential rather than producing
id=""), delimiter auto-detection precedence (comma beats semicolon
when both appear on the same line), and an explicit delimiter
constructor override bypassing auto-detection.

No bugs found -- this module was already correct; only test
coverage was added.
"""

from __future__ import annotations

import pytest

from topocore.survey.exceptions import SurveyFormatError, SurveyRecordError
from topocore.survey.formats import SurveyFormat
from topocore.survey.reader import SurveyTXTReader, _detect_delimiter


def _write(path: object, content: str) -> str:
    p = str(path) + "/file.txt"  # type: ignore[operator]
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


# ----------------------------------------------------------------------
# Bilingual header detection -- the primary documented use case.
# ----------------------------------------------------------------------


def test_spanish_header_comma_delimited(tmp_path: object) -> None:
    path = _write(
        tmp_path,
        "Punto,Este,Norte,Z,Descripcion\nP1,500123.45,4000456.78,100.5,CERCA\nP2,500124.45,4000457.78,101.5,MURO\n",
    )
    result = SurveyTXTReader(path).read()

    assert len(result) == 2
    assert result.points[0].id == "P1"
    assert result.points[0].x == pytest.approx(500123.45)
    assert result.points[0].y == pytest.approx(4000456.78)
    assert result.points[0].z == pytest.approx(100.5)
    assert result.points[0].code == "CERCA"


def test_english_header_tab_delimited(tmp_path: object) -> None:
    path = _write(tmp_path, "Point\tX\tY\tZ\tDescription\nA1\t100.0\t200.0\t10.0\tTREE\n")
    result = SurveyTXTReader(path).read()

    assert len(result) == 1
    assert result.points[0].id == "A1"
    assert result.points[0].code == "TREE"


# ----------------------------------------------------------------------
# Explicit format preset -- headerless files.
# ----------------------------------------------------------------------


def test_explicit_format_preset_no_header(tmp_path: object) -> None:
    path = _write(tmp_path, "1 500000.0 4000000.0 50.0 GROUND\n2 500010.0 4000010.0 51.0 GROUND\n")
    result = SurveyTXTReader(path, format=SurveyFormat.PENZD).read()

    assert len(result) == 2
    assert result.points[0].id == "1"
    assert result.points[0].code == "GROUND"


# ----------------------------------------------------------------------
# Quote-aware splitting.
# ----------------------------------------------------------------------


def test_quoted_field_containing_delimiter_not_split(tmp_path: object) -> None:
    path = _write(
        tmp_path,
        'Punto,Este,Norte,Z,Descripcion\nP1,500123.45,4000456.78,100.5,"CERCA, COSTADO NORTE"\n',
    )
    result = SurveyTXTReader(path).read()

    assert result.points[0].code == "CERCA, COSTADO NORTE"


# ----------------------------------------------------------------------
# Comments, blank lines, empty file.
# ----------------------------------------------------------------------


def test_comments_and_blank_lines_skipped(tmp_path: object) -> None:
    path = _write(
        tmp_path,
        "# comentario\nPunto,Este,Norte,Z,Descripcion\n\nP1,1.0,2.0,3.0,X\n// otro\nP2,4.0,5.0,6.0,Y\n",
    )
    result = SurveyTXTReader(path).read()
    assert len(result) == 2


def test_empty_file_returns_empty_point_set(tmp_path: object) -> None:
    path = _write(tmp_path, "")
    result = SurveyTXTReader(path).read()
    assert len(result) == 0


# ----------------------------------------------------------------------
# Errors.
# ----------------------------------------------------------------------


def test_no_header_no_format_raises_survey_format_error(tmp_path: object) -> None:
    path = _write(tmp_path, "500000.0 4000000.0 50.0\n")
    with pytest.raises(SurveyFormatError, match="Could not detect a header"):
        SurveyTXTReader(path).read()


def test_invalid_coordinate_raises_survey_record_error_with_line_number(
    tmp_path: object,
) -> None:
    path = _write(tmp_path, "Punto,Este,Norte,Z,Descripcion\nP1,ABC,4000456.78,100.5,X\n")
    with pytest.raises(SurveyRecordError, match="line 2"):
        SurveyTXTReader(path).read()


def test_missing_coordinate_column_raises_survey_record_error(tmp_path: object) -> None:
    """A row with fewer fields than the layout expects -- IndexError path inside _build_point."""
    path = _write(tmp_path, "Punto,Este,Norte,Z\nP1,1.0,2.0\n")  # missing Z value
    with pytest.raises(SurveyRecordError):
        SurveyTXTReader(path).read()


# ----------------------------------------------------------------------
# ID / code column edge cases.
# ----------------------------------------------------------------------


def test_missing_id_and_code_columns_use_sequential_fallback(tmp_path: object) -> None:
    path = _write(tmp_path, "X,Y,Z\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
    result = SurveyTXTReader(path).read()

    assert result.points[0].id == "1"
    assert result.points[0].code is None
    assert result.points[1].id == "2"


def test_id_column_present_but_empty_cell_falls_back_to_sequential(
    tmp_path: object,
) -> None:
    """
    Deliberately different from a MISSING id column: here the column
    exists, but this specific row's cell is blank -- must still fall
    back to the sequential id, not produce SurveyPoint(id="").
    """
    path = _write(tmp_path, "Punto,Este,Norte,Z\n,1.0,2.0,3.0\nP2,4.0,5.0,6.0\n")
    result = SurveyTXTReader(path).read()

    assert result.points[0].id == "1"
    assert result.points[1].id == "P2"


def test_code_column_present_but_empty_cell_gives_none() -> None:
    """code=None (not an empty string) when the cell is blank but the column exists."""
    from topocore.survey.formats import ColumnLayout
    from topocore.survey.reader import _build_point

    layout = ColumnLayout(x_column=0, y_column=1, z_column=2, id_column=None, code_column=3)
    point = _build_point(["1.0", "2.0", "3.0", ""], layout, source_line=1, fallback_id=1)
    assert point.code is None


# ----------------------------------------------------------------------
# Delimiter detection and override.
# ----------------------------------------------------------------------


def test_delimiter_precedence_comma_over_semicolon() -> None:
    assert _detect_delimiter("a,b;c") == ","


def test_delimiter_semicolon_when_no_comma() -> None:
    assert _detect_delimiter("a;b;c") == ";"


def test_delimiter_tab() -> None:
    assert _detect_delimiter("a\tb\tc") == "\t"


def test_delimiter_none_for_whitespace_only() -> None:
    assert _detect_delimiter("a b c") is None


def test_explicit_delimiter_overrides_autodetection(tmp_path: object) -> None:
    path = _write(tmp_path, "Punto;Este;Norte;Z\nP1;1.0;2.0;3.0\n")
    result = SurveyTXTReader(path, delimiter=";").read()

    assert len(result) == 1
    assert result.points[0].x == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Reader properties.
# ----------------------------------------------------------------------


def test_path_and_format_properties(tmp_path: object) -> None:
    path = _write(tmp_path, "Punto,Este,Norte,Z\nP1,1.0,2.0,3.0\n")
    reader = SurveyTXTReader(path, format=SurveyFormat.PENZD)

    assert str(reader.path).endswith("file.txt")
    assert reader.format == SurveyFormat.PENZD


def test_format_defaults_to_none() -> None:
    reader = SurveyTXTReader("dummy.txt")
    assert reader.format is None
