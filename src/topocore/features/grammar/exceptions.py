"""
topocore.features.grammar.exceptions
========================================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.exceptions import FeatureError


class FieldCodeGrammarError(FeatureError):
    """Raised when a grammar-syntax field code can't be parsed."""


__all__ = ["FieldCodeGrammarError"]
