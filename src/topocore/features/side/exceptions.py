"""
topocore.features.side.exceptions
====================================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.exceptions import FeatureError


class SideResolutionError(FeatureError):
    """
    Raised when `SideResolver.resolve()` itself cannot proceed --
    e.g. an explicit `attributes["side"]` value that is neither
    "left" nor "right".

    This is NOT raised when a side simply cannot be determined from
    the available evidence (no centerline, ambiguous candidates,
    near-zero cross product) -- that case produces
    `Side.UNKNOWN` with the appropriate `SideMethod`, not an
    exception. A missing or ambiguous answer is an expected,
    everyday outcome for real survey data, not an error.
    """


__all__ = ["SideResolutionError"]
