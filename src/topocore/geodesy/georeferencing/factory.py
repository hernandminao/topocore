"""
topocore.geodesy.georeferencing.factory
==========================================

`fit_georeferencing()`: the single orchestrator selecting among the
3 strategies (`HELMERT_3D`, `HELMERT_2D`, `TRANSLATION_ONLY`) based
purely on each fit's own mathematical determinacy -- never on
`len(controls)` alone (confirmed during this capability's own audit
that control point count does not determine this on its own: e.g. a
3D-degenerate configuration may or may not also be 2D-degenerate,
verified with a real counter-example).

The cascade, and why it stops trying at the first raised
`LinearizationInvalidError`
-----------------------------------------------------------------------
    HELMERT_3D (n >= 3)
        -> DegenerateGeometryError/InsufficientControlPointsError -> try HELMERT_2D
        -> LinearizationInvalidError -> propagate immediately, no further attempt
    HELMERT_2D (n >= 2)
        -> DegenerateGeometryError/InsufficientControlPointsError -> try TRANSLATION_ONLY
        -> LinearizationInvalidError -> propagate immediately, no further attempt
    TRANSLATION_ONLY (n >= 1)
        -> InsufficientControlPointsError (n == 0) -> propagate

A `LinearizationInvalidError` (a genuine mirror reflection or
similarly impossible-to-linearize configuration) is deliberately
NOT treated the same as a rank deficiency: falling back to a lower-
dimensional strategy would not fix a reflection, it would simply
compute a translation or 2D fit against fundamentally mismatched
data, silently producing a plausible-looking but meaningless
result -- masking the real problem instead of surfacing it. Rank
deficiencies are a legitimate "not enough independent information
yet" case that a lower-dimensional strategy genuinely can resolve;
a reflection is not.

The authorization gate
-------------------------
`HELMERT_3D` succeeding means the data itself fully determines all 7
parameters -- no authorization needed, `warning=None`. `HELMERT_2D`
and `TRANSLATION_ONLY` both assume parameters the data cannot
determine (`rx`/`ry` for 2D; `rx`/`ry`/`rz`/`scale` for translation)
-- both require `GeoreferencingOptions.accept_underconstrained=True`,
raising `UnderconstrainedGeoreferencingError` otherwise. This is
never inferred from control point count; it is always the caller's
own explicit decision (see `GeoreferencingOptions`'s own docstring).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from topocore.geodesy.georeferencing.control_point import ControlPoint
from topocore.geodesy.georeferencing.exceptions import (
    ControlPointsTooCloseError,
    DegenerateGeometryError,
    InsufficientControlPointsError,
    UnderconstrainedGeoreferencingError,
)
from topocore.geodesy.georeferencing.helmert_2d import fit_helmert_2d
from topocore.geodesy.georeferencing.helmert_fit import HelmertFitResult, fit_helmert_3d
from topocore.geodesy.georeferencing.options import GeoreferencingOptions
from topocore.geodesy.georeferencing.result import (
    GeoreferencingResult,
    GeoreferencingStrategy,
)
from topocore.geodesy.georeferencing.single_control import fit_translation_only

_TRANSLATION_ONLY_WARNING = (
    "Only {n} control point(s) available: translation-only. tx/ty/tz were estimated; "
    "rx=ry=rz=0.0 and scale=0.0 ppm were ASSUMED, not measured, because a single "
    "translation (3 unknowns) cannot determine rotation or scale at all. "
    "Precision degrades with distance from the control point(s)."
)
_HELMERT_2D_WARNING = (
    "Only {n} control points available: 2D Helmert. tx/ty/tz/rz/scale were estimated "
    "from the control points' own XY relationship; rx=ry=0.0 (rotation about the "
    "horizontal axes) was ASSUMED, not measured -- the 3D orientation of the surface "
    "cannot be independently determined from a purely 2D configuration."
)
_HELMERT_2D_DEGENERATE_3D_WARNING = (
    "The {n} control points given do not provide enough independent information for a "
    "full 3D Helmert fit; falling back to 2D Helmert. tx/ty/tz/rz/scale were estimated; "
    "rx=ry=0.0 was ASSUMED, not measured."
)
_TRANSLATION_ONLY_DEGENERATE_2D_WARNING = (
    "The {n} control points given do not provide enough independent information for "
    "either a 3D or a 2D Helmert fit (their own XY projection is itself degenerate, e.g. "
    "all sharing the same horizontal position); falling back to translation-only. "
    "tx/ty/tz were estimated; rx=ry=rz=0.0 and scale=0.0 ppm were ASSUMED, not measured."
)
_TRANSLATION_ONLY_DEGENERATE_2D_ONLY_WARNING = (
    "The {n} control points given do not provide enough independent information for a "
    "2D Helmert fit (their own XY projection is degenerate, e.g. sharing the same "
    "horizontal position -- a genuine 3D fit was never attempted, since that requires at "
    "least 3 control points); falling back to translation-only. tx/ty/tz were estimated; "
    "rx=ry=rz=0.0 and scale=0.0 ppm were ASSUMED, not measured."
)


def _to_georeferencing_result(
    fit_result: HelmertFitResult,
    strategy: GeoreferencingStrategy,
    control_count: int,
    warning: str | None,
) -> GeoreferencingResult:
    # A single control point under TRANSLATION_ONLY has zero
    # redundancy (3 unknowns, 3 equations) -- HelmertFitResult.rms is
    # trivially 0.0 in that specific case, which GeoreferencingResult
    # reports as None ("not applicable") rather than a real number,
    # per this module's own docstring and GeoreferencingResult's own.
    rms: float | None = fit_result.rms
    if strategy is GeoreferencingStrategy.TRANSLATION_ONLY and control_count == 1:
        rms = None

    return GeoreferencingResult(
        strategy=strategy,
        parameters=fit_result.parameters,
        residuals=fit_result.residuals,
        rms=rms,
        control_count=control_count,
        warning=warning,
    )


def _check_minimum_control_distance(controls: Sequence[ControlPoint], minimum_control_distance: float | None) -> None:
    if minimum_control_distance is None:
        return

    for i in range(len(controls)):
        for j in range(i + 1, len(controls)):
            xi, yi, _zi = controls[i].source
            xj, yj, _zj = controls[j].source
            distance_xy = math.hypot(xj - xi, yj - yi)
            if distance_xy < minimum_control_distance:
                raise ControlPointsTooCloseError(
                    f"fit_georeferencing(): control points '{controls[i].id}' and '{controls[j].id}' "
                    f"are only {distance_xy:.4g} apart horizontally, below the configured "
                    f"minimum_control_distance of {minimum_control_distance:.4g}."
                )


def fit_georeferencing(controls: Sequence[ControlPoint], options: GeoreferencingOptions) -> GeoreferencingResult:
    """
    Select and run whichever of `HELMERT_3D`/`HELMERT_2D`/
    `TRANSLATION_ONLY` the control points' own mathematical rank
    supports -- see this module's own docstring for the exact
    cascade and authorization policy.

    Raises
    ------
    InsufficientControlPointsError
        If `controls` is empty.
    UnderconstrainedGeoreferencingError
        If only an under-determined strategy (`HELMERT_2D` or
        `TRANSLATION_ONLY`) is available and
        `options.accept_underconstrained` is not `True`.
    ControlPointsTooCloseError
        If `options.minimum_control_distance` is set and violated.
    LinearizationInvalidError
        Propagated directly from whichever fit raised it -- never
        caught to attempt a lower-dimensional fallback (see this
        module's own docstring for why).
    """
    if len(controls) == 0:
        raise InsufficientControlPointsError("fit_georeferencing() requires at least 1 control point, got 0.")

    _check_minimum_control_distance(controls, options.minimum_control_distance)

    n = len(controls)

    if n >= 3:
        try:
            fit_3d = fit_helmert_3d(controls)
            return _to_georeferencing_result(fit_3d, GeoreferencingStrategy.HELMERT_3D, n, warning=None)
        except (InsufficientControlPointsError, DegenerateGeometryError):
            pass  # fall through to HELMERT_2D below

    if n >= 2:
        try:
            fit_2d = fit_helmert_2d(controls)
        except (InsufficientControlPointsError, DegenerateGeometryError):
            fit_2d = None

        if fit_2d is not None:
            if not options.accept_underconstrained:
                raise UnderconstrainedGeoreferencingError(
                    "fit_georeferencing(): only an under-determined HELMERT_2D fit is available "
                    f"for these {n} control points (rx/ry cannot be determined). Set "
                    "GeoreferencingOptions(accept_underconstrained=True) to explicitly authorize this."
                )
            warning_template = _HELMERT_2D_DEGENERATE_3D_WARNING if n >= 3 else _HELMERT_2D_WARNING
            return _to_georeferencing_result(
                fit_2d,
                GeoreferencingStrategy.HELMERT_2D,
                n,
                warning=warning_template.format(n=n),
            )

    fit_translation = fit_translation_only(controls)
    if not options.accept_underconstrained:
        raise UnderconstrainedGeoreferencingError(
            f"fit_georeferencing(): only an under-determined TRANSLATION_ONLY fit is available for "
            f"these {n} control point(s) (rotation and scale cannot be determined). Set "
            "GeoreferencingOptions(accept_underconstrained=True) to explicitly authorize this."
        )
    if n >= 3:
        warning_template = _TRANSLATION_ONLY_DEGENERATE_2D_WARNING  # both 3D and 2D were genuinely attempted
    elif n == 2:
        warning_template = (
            _TRANSLATION_ONLY_DEGENERATE_2D_ONLY_WARNING  # only 2D was attempted -- n < 3 never tries 3D at all
        )
    else:
        warning_template = _TRANSLATION_ONLY_WARNING  # n == 1 -- no fallback happened, this is the base case
    return _to_georeferencing_result(
        fit_translation,
        GeoreferencingStrategy.TRANSLATION_ONLY,
        n,
        warning=warning_template.format(n=n),
    )


__all__ = ["fit_georeferencing"]
