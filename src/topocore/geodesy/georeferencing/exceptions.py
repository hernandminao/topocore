"""
topocore.geodesy.georeferencing.exceptions
=============================================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geodesy.exceptions import GeodesyError


class GeoreferencingError(GeodesyError):
    """Base exception for control-point-based georeferencing errors."""


class InsufficientControlPointsError(GeoreferencingError):
    """
    Raised when fewer than 3 control points are given. A 3D Helmert
    similarity fit has 7 unknowns; 3 non-collinear points give the
    minimum 9 equations needed to solve for them.
    """


class DegenerateGeometryError(GeoreferencingError):
    """
    Raised when the control points, though numerous enough, do not
    provide 7 independent equations for the fit -- confirmed
    directly this happens for collinear points (and any other
    configuration collapsing the design matrix's own rank below 7),
    never silently accepted as if it produced a meaningful solution.
    """


class LinearizationInvalidError(GeoreferencingError):
    """
    Raised when the fitted rotation exceeds the range where the
    small-angle linearization this fit relies on remains valid.

    Confirmed directly: a genuine mirror reflection (e.g. source
    data in a left-handed convention fit against right-handed
    targets) is NOT representable by a small rotation + scale at
    all -- Helmert/Bursa-Wolf similarity transforms are proper
    (determinant +1) by construction, and cannot represent an
    improper (reflective, determinant -1) mapping. Forcing the
    linear least-squares system to fit one anyway does not fail
    outright -- it silently produces a "solution" with huge,
    physically meaningless rotation and scale values (e.g. tens of
    degrees and hundreds of thousands of ppm, confirmed directly with
    a real reflected-axis example) that a caller could otherwise
    mistake for a valid, if poor, fit.

    The threshold (3600 arc-seconds = 1 degree) is deliberately
    generous for genuine geodetic/survey use: confirmed directly
    that the small-angle approximation's own relative error at this
    threshold is only ~0.005% (negligible), and real control-point
    Helmert fits between nearby, correctly-corresponded datums
    essentially never approach even a small fraction of this --
    exceeding it is a strong signal of a genuine problem with the
    input (a reflection, badly mismatched correspondences, ...), not
    merely "a somewhat large but still valid" rotation.
    """


class UnderconstrainedGeoreferencingError(GeoreferencingError):
    """
    Raised by `fit_georeferencing()` when the only strategy the
    control points can support (`TRANSLATION_ONLY` or `HELMERT_2D`)
    requires assuming parameters the data itself cannot determine
    (e.g. rotation/scale from a single control point), and the
    caller has not explicitly set
    `GeoreferencingOptions.accept_underconstrained=True`.

    This is deliberately never inferred from control point count
    alone: TopoCore never silently assumes a value for a physically
    real, unmeasured quantity (confirmed elsewhere in this project as
    a hard rule, following the exact real, demonstrated harm of doing
    so once already -- `Workflow.transform_crs()` producing infinite
    coordinates from a `crs=None` origin it silently assumed matched
    the transformer's own source CRS). A caller who understands and
    accepts this must say so explicitly; the resulting fit's own
    `GeoreferencingResult.warning` then documents exactly what was
    assumed, as a record of an authorized decision -- not as after-
    the-fact mitigation for one made silently.
    """


class ControlPointsTooCloseError(GeoreferencingError):
    """
    Raised when `GeoreferencingOptions.minimum_control_distance` is
    set and at least one pair of control points has a horizontal
    (XY-only) separation between their own `source` coordinates
    below that threshold. Never raised when
    `minimum_control_distance` is `None` (no check is performed at
    all in that case, not a default threshold).
    """


__all__ = [
    "ControlPointsTooCloseError",
    "DegenerateGeometryError",
    "GeoreferencingError",
    "InsufficientControlPointsError",
    "LinearizationInvalidError",
    "UnderconstrainedGeoreferencingError",
]
