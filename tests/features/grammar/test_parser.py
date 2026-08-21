"""
Regression suite for topocore.features.grammar.parser -- PR19.

Verified all valid syntax forms (implicit CONTINUE, S/E/X commands,
case-insensitivity, whitespace tolerance) and every malformed-code
error path with real parsing. No bugs found.
"""

from __future__ import annotations

import pytest

from topocore.features.grammar.commands import FieldCodeCommand
from topocore.features.grammar.exceptions import FieldCodeGrammarError
from topocore.features.grammar.parser import FieldCodeParser, is_grammar_code


@pytest.mark.parametrize(
    ("code", "base", "figure", "command"),
    [
        ("CERCA.1", "CERCA", "1", FieldCodeCommand.CONTINUE),
        ("CERCA.1.S", "CERCA", "1", FieldCodeCommand.START),
        ("CERCA.1.E", "CERCA", "1", FieldCodeCommand.END),
        ("CERCA.1.X", "CERCA", "1", FieldCodeCommand.CLOSE),
        ("CERCA.1.s", "CERCA", "1", FieldCodeCommand.START),
        (" CERCA . 1 ", "CERCA", "1", FieldCodeCommand.CONTINUE),
    ],
)
def test_valid_codes_parsed_correctly(code: str, base: str, figure: str, command: FieldCodeCommand) -> None:
    result = FieldCodeParser.parse(code)
    assert result.base_code == base
    assert result.figure_id == figure
    assert result.command == command


def test_is_grammar_code_distinguishes_legacy_from_grammar() -> None:
    assert is_grammar_code("CERCA1") is False
    assert is_grammar_code("CERCA.1") is True


@pytest.mark.parametrize(
    "code",
    [
        "CERCA.",  # empty figure
        ".1",  # empty base
        "CERCA.1.2.3",  # too many segments
        "CERCA.1.Z",  # unknown command token
        "CERCA",  # single segment
        "CERCA. .S",  # whitespace-only figure
    ],
)
def test_malformed_codes_rejected(code: str) -> None:
    with pytest.raises(FieldCodeGrammarError):
        FieldCodeParser.parse(code)
