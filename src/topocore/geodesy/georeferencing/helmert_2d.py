"""
topocore.geodesy.georeferencing.helmert_2d
=============================================

`fit_helmert_2d()`: the `HELMERT_2D` strategy -- `tx`/`ty`/`s`/`rz`
estimated by linear least-squares from the control points' own XY
relationship, `tz` separately estimated (mean of `target.z -
source.z`, matching `fit_translation_only()`'s own approach exactly,
since Z has no rotation/scale coupling to worry about here);
`rx`/`ry` are never estimated, always assumed `0.0`.

Its own, independent system -- never derived by "turning off" 3
parameters from `fit_helmert_3d()`
------------------------------------------------------------------
Confirmed directly, during this capability's own audit: a 3D
Helmert fit's own degeneracy (`rank_3d < 7`) does NOT imply this 2D
system is also degenerate, nor the reverse -- a real, concrete
counter-example was found (control points collinear along a purely
vertical line: `rank_3d = 6` AND `rank_2d = 2`, both degenerate,
while collinear points along a horizontal or diagonal line give
`rank_3d = 6` but `rank_2d = 4`, fully determined). This is exactly
why this function builds and solves its own, independent 4x4 (XY)
design matrix rather than trying to reuse or reduce
`fit_helmert_3d()`'s own 7-parameter system -- the two fits' own
degeneracy conditions are genuinely different problems, verified
directly rather than assumed.

Scope: assumes horizontal scale is the only relevant scale error
-------------------------------------------------------------------
This model assumes Z's own error is purely a constant offset
(`tz`), never a scale factor -- correct for a total station (angles
and slope distance reduced to horizontal, so horizontal and vertical
measurement error do not generally share the same scale error), but
NOT necessarily correct for other instruments or workflows where the
same linear measuring error genuinely affects Z proportionally too
(e.g. a tape measure used consistently, without correction, for both
horizontal and vertical distances). This is a deliberate, stated
scope for the total-station case this strategy primarily exists for,
not a general claim that vertical scale error can never occur -- a
survey where it can should not use `HELMERT_2D` to characterize its
own Z error.

Same position-vector formula, linearized, as the 3D fit -- restricted
to the XY plane:

    Xt = Xs + tx + s*Xs - rz*Ys
    Yt = Ys + ty + s*Ys + rz*Xs

(this is `fit_helmert_3d()`'s own formula with `rx=ry=0` -- verified
directly this is algebraically identical to what remains of that
formula's own X/Y equations once those 2 terms are dropped, so a
`HelmertParameters` produced here means the same physical
transformation as one from the 3D fit would, for whatever it does
determine).

**Z is handled separately, and is NOT the same as `apply_helmert_3d()`
with `rx=ry=0`.** This fit's own Z model is `Z' = Z + tz` alone --
`tz` is the mean of `target.z - source.z` across all controls, with
no scale term at all. `apply_helmert_3d()`'s own formula, even with
`rx=ry=0`, still includes `+ s*Zs` (confirmed directly: this created
a real, if often small, discrepancy when this module's own residual
computation used to call it) -- this module never calls
`apply_helmert_3d()` for this reason, computing its own residuals
directly against its own `Z' = Z + tz` model instead.

A reflection is caught the same way as in the 3D fit: see
`LinearizationInvalidError`'s own docstring; the identical rotation-
magnitude check applies here too, since the same small-angle
linearization breakdown applies to `rz` alone just as it does to
`rx`/`ry`/`rz` together.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from math import degrees

import numpy as np

from topocore.geodesy.georeferencing.control_point import ControlPoint
from topocore.geodesy.georeferencing.exceptions import (
    DegenerateGeometryError,
    InsufficientControlPointsError,
    LinearizationInvalidError,
)
from topocore.geodesy.georeferencing.helmert_fit import (
    HelmertFitResult,
    Residual,
)
from topocore.geodesy.helmert import HelmertParameters

_MIN_CONTROL_POINTS = 2
_REQUIRED_RANK = 4
_MAX_VALID_ROTATION_ARCSEC = 3600.0


def fit_helmert_2d(controls: Sequence[ControlPoint]) -> HelmertFitResult:
    """
    Estimate a 2D Helmert similarity transformation (`tx`, `ty`, `tz`,
    `rz`, `scale`; `rx`/`ry` always assumed `0.0`) from `controls`.

    Raises
    ------
    InsufficientControlPointsError
        If fewer than 2 control points are given.
    DegenerateGeometryError
        If the control points' own XY design matrix does not reach
        rank 4 -- confirmed directly this happens when all control
        points share the same XY projection (e.g. a purely vertical
        line of points), regardless of how many are given or how
        different their Z values are.
    LinearizationInvalidError
        If the fitted `rz` exceeds the same 1-degree bound used by
        `fit_helmert_3d()` -- see that exception's own docstring.
    """
    if len(controls) < _MIN_CONTROL_POINTS:
        raise InsufficientControlPointsError(
            f"fit_helmert_2d() requires at least {_MIN_CONTROL_POINTS} control points "
            f"(the XY fit has 4 unknowns), got {len(controls)}."
        )

    n = len(controls)
    design_matrix = np.zeros((2 * n, 4))
    observations = np.zeros(2 * n)

    for i, control in enumerate(controls):
        xs, ys, _zs = control.source
        xt, yt, _zt = control.target

        design_matrix[2 * i, :] = [1, 0, xs, -ys]
        design_matrix[2 * i + 1, :] = [0, 1, ys, xs]

        observations[2 * i : 2 * i + 2] = [xt - xs, yt - ys]

    solution, _sum_sq_residuals, rank, _singular_values = np.linalg.lstsq(design_matrix, observations, rcond=None)

    if rank < _REQUIRED_RANK:
        raise DegenerateGeometryError(
            f"fit_helmert_2d(): the {n} control points given do not provide {_REQUIRED_RANK} "
            f"independent equations in their own XY projection (design matrix rank is {rank}) -- "
            "confirmed this happens when all control points share the same XY position, regardless "
            "of their Z values. Provide control points with genuine XY spread."
        )

    tx, ty, s, rz = solution
    rz_arcsec = degrees(rz) * 3600.0

    if abs(rz_arcsec) > _MAX_VALID_ROTATION_ARCSEC:
        raise LinearizationInvalidError(
            f"fit_helmert_2d(): the fitted rotation (rz={rz_arcsec:.1f} arcsec) exceeds "
            f"{_MAX_VALID_ROTATION_ARCSEC:.0f} arcsec (1 degree) -- beyond the range where this "
            "fit's own small-angle linearization remains valid. This typically means the control "
            "points cannot be related by a genuine small-rotation 2D similarity transform at all "
            "(e.g. a mirror reflection between source and target)."
        )

    tz_deltas = [control.target[2] - control.source[2] for control in controls]
    tz = float(np.mean(tz_deltas))

    parameters = HelmertParameters(
        tx=float(tx),
        ty=float(ty),
        tz=tz,
        rz=rz_arcsec,
        scale=float(s) * 1e6,
    )

    residuals = []
    squared_magnitudes = []
    for control in controls:
        # Deliberately NOT apply_helmert_3d(): that function's own
        # formula applies `scale` to Z too (zt = zs + tz + s*zs -
        # ry*xs + rx*ys), which -- even with rx=ry=0 -- is NOT the
        # same model this fit actually solved for. This fit's own Z
        # model is `Z' = Z + tz` alone, with no scale term at all
        # (confirmed directly: reusing apply_helmert_3d() here
        # contaminated residuals/RMS with a spurious s*zs term never
        # part of what was actually fit, for any nonzero scale and
        # nonzero Z). Applying the fitted X/Y formula directly here
        # keeps residuals faithful to this model's own definition.
        xs, ys, zs = control.source
        xt = xs + parameters.tx + s * xs - rz * ys
        yt = ys + parameters.ty + s * ys + rz * xs
        zt = zs + parameters.tz

        dx = control.target[0] - xt
        dy = control.target[1] - yt
        dz = control.target[2] - zt
        magnitude = float(np.sqrt(dx**2 + dy**2 + dz**2))
        residuals.append(Residual(control_id=control.id, dx=dx, dy=dy, dz=dz, magnitude=magnitude))
        squared_magnitudes.append(magnitude**2)

    rms = float(np.sqrt(np.mean(squared_magnitudes)))

    return HelmertFitResult(parameters=parameters, residuals=tuple(residuals), rms=rms)


__all__ = ["fit_helmert_2d"]
