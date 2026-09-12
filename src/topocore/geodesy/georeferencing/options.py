"""
topocore.geodesy.georeferencing.options
==========================================

`GeoreferencingOptions`: the caller-controlled policy for
`fit_georeferencing()` -- specifically, the explicit authorization
required before a mathematically under-determined strategy
(`TRANSLATION_ONLY`, `HELMERT_2D`, or a `HELMERT_2D`/`TRANSLATION_ONLY`
fallback from a degenerate higher-rank attempt) is allowed to run at
all, rather than being rejected outright.

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
class GeoreferencingOptions:
    """
    Parameters
    ----------
    accept_underconstrained
        `False` by default: `fit_georeferencing()` raises rather than
        run any strategy whose parameters are not fully determined by
        the control points given (see
        `topocore.geodesy.georeferencing.exceptions.UnderconstrainedGeoreferencingError`).
        Set `True` to explicitly authorize this -- the caller is
        confirming they understand and accept that some parameters
        will be assumed, not measured. This is never inferred from
        control point count alone; it is always the caller's own
        explicit decision.
    minimum_control_distance
        `None` (the default) means no proximity check is performed
        at all -- not "use some default threshold". Set to a positive
        number to require every pair of control points to be
        separated by at least that much, measured as the horizontal
        (XY-plane only, never including Z) distance between their
        own `source` coordinates, in the same linear units as those
        coordinates. Relevant specifically to `HELMERT_2D`'s own
        stability: its rotation/scale are derived entirely from the
        XY relationship between control points, which becomes
        numerically unstable as that separation shrinks toward zero
        (irrespective of how far apart the points are in Z).
    """

    accept_underconstrained: bool = False
    minimum_control_distance: float | None = None


__all__ = ["GeoreferencingOptions"]
