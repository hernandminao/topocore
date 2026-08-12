"""
topocore.features._code_utils
=================================

Tiny shared utility, extracted from `feature_builder.py` so both it
and `features.grammar` can use the same base-code stripping rule
without a circular import (`feature_builder.py` depends on
`grammar/` for the opt-in grammar mode; `grammar/` must not depend
back on `feature_builder.py`).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations


def base_code(code: str) -> str:
    """
    Strip trailing digits from a field code (``"CERCA1"`` ->
    ``"CERCA"``). If the code is entirely digits, returns it
    unchanged rather than an empty string.
    """
    stripped = code.rstrip("0123456789")
    return stripped if stripped else code


__all__ = ["base_code"]
