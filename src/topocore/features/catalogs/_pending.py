"""
topocore.features.catalogs._pending
=====================================

Explicit registry of survey codes intentionally not yet migrated.

This can't be derived automatically from ALL_CODES -- a code that's
absent from a catalog is, by definition, invisible to any code that
only inspects ALL_CODES. Keeping this list maintained by hand is
what lets catalog_audit report "pending" instead of silently having
no idea ROCAARBOL exists at all.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PendingCode:
    code: str
    catalog: str
    reason: str


PENDING_CODES: tuple[PendingCode, ...] = (
    PendingCode(
        code="ROCAARBOL",
        catalog="vegetation",
        reason='Declared name ("Tree on Rock") does not disambiguate what it marks '
        "(the rock, the tree, or a composite feature) -- deferred until confirmed "
        "against real field usage rather than guessed from the name alone.",
    ),
)

__all__ = ["PENDING_CODES", "PendingCode"]
