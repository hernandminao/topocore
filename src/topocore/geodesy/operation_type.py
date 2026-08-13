"""
topocore.geodesy.operation_type
===================================

Categorical vocabulary for the kind of coordinate operation a
`CoordinateOperation` describes. Deliberately limited to the
operation kinds PR18A.2 actually built a parameter model for
(`HelmertParameters`, `GridShift`) plus `IDENTITY` (no parameters at
all) -- no `MOLODENSKY` or similar entries with nothing behind them.
New values are added alongside their parameter model, in the same
PR, never speculatively.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import StrEnum


class OperationType(StrEnum):
    """
    IDENTITY
        No transformation parameters -- e.g. same datum, only a
        projection or axis-order change pyproj handles directly.
    HELMERT
        Backed by `HelmertParameters` (7 or 14 parameter form).
    GRID_SHIFT
        Backed by `GridShift`.
    """

    IDENTITY = "identity"
    HELMERT = "helmert"
    GRID_SHIFT = "grid_shift"


__all__ = ["OperationType"]
