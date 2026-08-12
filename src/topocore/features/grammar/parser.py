"""
topocore.features.grammar.parser
====================================

Parses the optional field-code grammar syntax::

    BASE.FIGURE          -> CONTINUE (implicit)
    BASE.FIGURE.S        -> START
    BASE.FIGURE.E        -> END
    BASE.FIGURE.X        -> CLOSE

The separator is a module constant (``.`` by default), not hardcoded
throughout the parser logic, so changing it later doesn't mean
rewriting the regex/split logic -- only the constant.

A code is grammar-syntax if and only if it contains the separator.
This is what makes ``"CERCA1"`` (legacy: a plain code, digit
ignored) and ``"CERCA.1"`` (grammar: base "CERCA", figure "1",
implicit CONTINUE) unambiguous -- no heuristics needed to guess
which convention the surveyor meant.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.features.grammar.commands import FieldCodeCommand
from topocore.features.grammar.exceptions import FieldCodeGrammarError

#: The field-code segment separator. A module constant, not a
#: literal scattered through the parser, so it can be revisited
#: without touching parsing logic.
SEPARATOR = "."

_COMMAND_TOKENS: dict[str, FieldCodeCommand] = {
    "S": FieldCodeCommand.START,
    "E": FieldCodeCommand.END,
    "X": FieldCodeCommand.CLOSE,
}


def is_grammar_code(code: str) -> bool:
    """Whether `code` uses the grammar syntax (contains SEPARATOR) rather than the legacy convention."""
    return SEPARATOR in code


@dataclass(frozen=True, slots=True)
class ParsedFieldCode:
    """
    Parameters
    ----------
    raw_code
        The original, unparsed code string.
    base_code
        The feature-code part, before the separator (looked up in
        `FeatureCodeRegistry` exactly like a legacy base code).
    figure_id
        Figure identifier. ``str``, not ``int`` -- V1 only ever
        parses digits into it, but nothing downstream (the
        assembler, `FeatureBuilder`) needs to know that, so a future
        grammar extension isn't blocked by this type.
    command
        The parsed instruction. ``FieldCodeCommand.CONTINUE`` when
        the code has no third segment.
    """

    raw_code: str
    base_code: str
    figure_id: str | None
    command: FieldCodeCommand


class FieldCodeParser:
    """Stateless parser: the only thing that translates grammar syntax into `ParsedFieldCode`."""

    @staticmethod
    def parse(code: str) -> ParsedFieldCode:
        """
        Raises
        ------
        FieldCodeGrammarError
            If `code` doesn't have exactly 2 or 3 SEPARATOR-delimited
            segments, if the base or figure segment is empty, or if
            a 3-segment code's command token isn't a recognized one.
        """
        parts = code.split(SEPARATOR)

        if len(parts) == 2:
            base, figure = parts
            command = FieldCodeCommand.CONTINUE
        elif len(parts) == 3:
            base, figure, command_token = parts
            token = command_token.strip().upper()
            if token not in _COMMAND_TOKENS:
                raise FieldCodeGrammarError(
                    f"Unknown command token '{command_token}' in code '{code}'. "
                    f"Must be one of {sorted(_COMMAND_TOKENS)}, or omit the third "
                    f"segment entirely for CONTINUE."
                )
            command = _COMMAND_TOKENS[token]
        else:
            raise FieldCodeGrammarError(
                f"Malformed grammar code '{code}': expected "
                f"'BASE{SEPARATOR}FIGURE' or 'BASE{SEPARATOR}FIGURE{SEPARATOR}COMMAND', "
                f"got {len(parts)} segments."
            )

        base = base.strip()
        figure = figure.strip()

        if not base:
            raise FieldCodeGrammarError(f"Malformed grammar code '{code}': base code is empty.")
        if not figure:
            raise FieldCodeGrammarError(f"Malformed grammar code '{code}': figure id is empty.")

        return ParsedFieldCode(raw_code=code, base_code=base, figure_id=figure, command=command)


__all__ = ["SEPARATOR", "FieldCodeParser", "ParsedFieldCode", "is_grammar_code"]
