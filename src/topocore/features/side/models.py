"""
topocore.features.side.models
================================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import StrEnum


class Side(StrEnum):
    """
    Lateral position of a linear feature (e.g. a pavement edge)
    relative to a reference centerline's own direction of travel.

    Never a `FeatureType` -- see the module docstring of
    `topocore.features.side.resolver` for why laterality is a
    property of the Feature *instance* (stored in
    `Feature.attributes`), not a distinct feature type.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class SideMethod(StrEnum):
    """
    How a `Side` value was determined -- kept separate from `Side`
    itself so "not yet resolved" and "resolved, but ambiguous" are
    both distinguishable from a confident answer, for auditing.
    """

    #: `attributes["side"]` was already present on the input Feature
    #: (survey-provided or set by an earlier stage); used as-is.
    EXPLICIT = "explicit"

    #: Computed from a single, unambiguous nearest CENTERLINE.
    GEOMETRIC = "geometric"

    #: Two or more CENTERLINE candidates were comparably close, or
    #: the computed cross product was within tolerance of zero
    #: (the edge point lies effectively ON the centerline's own
    #: line) -- resolved to UNKNOWN rather than guessing.
    AMBIGUOUS = "ambiguous"

    #: No CENTERLINE candidate was found within the configured
    #: maximum distance (including the case of zero CENTERLINE
    #: features in the collection at all).
    UNKNOWN = "unknown"


__all__ = ["Side", "SideMethod"]
