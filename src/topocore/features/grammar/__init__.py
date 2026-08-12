"""
topocore.features.grammar
=============================

Optional field-code grammar (V1: START/CONTINUE/END/CLOSE, figure
identity, simultaneous open figures). Opt-in only -- the legacy
`_base_code()`/`_group_runs()` convention in `feature_builder.py`
remains the default and is completely unchanged by this package's
existence.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.grammar.assembler import (
    AssembledDiagnostic,
    AssembledDiagnosticReason,
    AssembledFigure,
    AssembleResult,
    assemble,
)
from topocore.features.grammar.commands import FieldCodeCommand
from topocore.features.grammar.exceptions import FieldCodeGrammarError
from topocore.features.grammar.parser import (
    SEPARATOR,
    FieldCodeParser,
    ParsedFieldCode,
    is_grammar_code,
)

__all__ = [
    "SEPARATOR",
    "AssembleResult",
    "AssembledDiagnostic",
    "AssembledDiagnosticReason",
    "AssembledFigure",
    "FieldCodeCommand",
    "FieldCodeGrammarError",
    "FieldCodeParser",
    "ParsedFieldCode",
    "assemble",
    "is_grammar_code",
]
