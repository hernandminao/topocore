"""
topocore.geodesy.georeferencing.helmert_fit
==============================================

The pure mathematical core of control-point-based georeferencing:
`fit_helmert_3d()` estimates a 3D Helmert similarity transformation
(translation + small rotation + scale) from `ControlPoint`
correspondences via linear least-squares; `apply_helmert_3d()`
applies an already-known one directly to a Cartesian point.

Deliberately independent of CRS/PROJ/geocentric coordinates
-------------------------------------------------------------
This is a plain Cartesian-to-Cartesian fit -- `source`/`target` are
arbitrary (X, Y, Z) numbers, never interpreted as geographic
degrees. This is what makes it usable for a real total-station
survey (local, unitless station coordinates) fit against known
target coordinates in a projected CRS, unlike
`topocore.geodesy.CoordinateTransformer`'s own HELMERT support
(confirmed during this capability's own audit: that path requires
BOTH source and target CRS to be geographic, and internally
round-trips through geocentric coordinates via a PROJ pipeline --
architecturally correct for geodetic datum realizations, but not
applicable here).

The exact convention, fixed once, here
------------------------------------------
Position-vector convention (EPSG operation method 9606, the Bursa-
Wolf formula), linearized for small rotations -- the SAME convention
`CoordinateTransformer`'s own PROJ-based HELMERT path already uses
(confirmed directly, its own pipeline is built with
`+convention=position_vector`), so a `HelmertParameters` produced
here means the same physical transformation as one built by hand for
that other path, even though the two are never used together in this
first slice:

    Xt = Xs + tx + s*Xs - rz*Ys + ry*Zs
    Yt = Ys + ty + s*Ys + rz*Xs - rx*Zs
    Zt = Zs + tz + s*Zs - ry*Xs + rx*Ys

where `tx`/`ty`/`tz` are in the same linear units as the input
coordinates, `rx`/`ry`/`rz` are small rotations in radians (converted
to/from `HelmertParameters`'s own arc-second fields at the
boundary), and `s` is a dimensionless scale factor (converted to/from
`HelmertParameters`'s own ppm field at the boundary). This is a
genuine linear system in the 7 unknowns for known correspondences --
`fit_helmert_3d()` solves it directly via `numpy.linalg.lstsq`, no
iterative nonlinear optimization needed, matching the same
`numpy.linalg.lstsq` least-squares approach already established
elsewhere in this codebase (`topocore.processing.registration.point_to_plane`).

`apply_helmert_3d()` uses this exact same formula (forward
direction) -- confirmed directly there is no discrepancy between
what the fit solves for and what applying the result produces,
verified by round-tripping known parameters through both functions.

Degenerate geometry is detected via the design matrix's own rank
-------------------------------------------------------------------
Confirmed directly: fewer than 3 control points structurally cannot
reach rank 7 (a 3-point fit gives exactly 9 equations for 7
unknowns, the minimum possible), and collinear points reach at most
rank 6 regardless of point count (verified directly: 4 collinear
points give rank 6, never 7). Both cases are caught by the identical
mechanism -- checking `numpy.linalg.lstsq`'s own returned rank -- not
two separate ad-hoc heuristics.

**Coplanar (but non-collinear) points are confirmed NOT degenerate**
for this fit, contrary to intuition carried over from other
geometric problems (like plane fitting). Confirmed directly with a
real example: 4 points sharing Z=0 but spread in X/Y (a square) give
rank 7 and correctly recover a genuine rotation about X, since the Z
equation's own coefficients for rx/ry depend on X/Y (which vary),
not on Z itself.

A reflection is caught separately, via `LinearizationInvalidError`
-------------------------------------------------------------------
Confirmed directly, and importantly NOT the same failure mode as
degenerate geometry: fitting a genuine mirror reflection (source and
target related by an axis flip, determinant -1) does not produce a
rank deficiency at all -- the linear system solves "successfully",
but produces wildly unphysical rotation and scale values (tens of
degrees, hundreds of thousands of ppm) because a proper (determinant
+1) similarity transform cannot represent an improper one, and the
small-angle linearization breaks down catastrophically trying. Caught
by checking the fitted rotation against a 1-degree bound -- see
`LinearizationInvalidError`'s own docstring for the full rationale
and why this threshold was chosen.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import degrees, radians

import numpy as np

from topocore.geodesy.georeferencing.control_point import ControlPoint
from topocore.geodesy.georeferencing.exceptions import (
    DegenerateGeometryError,
    InsufficientControlPointsError,
    LinearizationInvalidError,
)
from topocore.geodesy.helmert import HelmertParameters

_MIN_CONTROL_POINTS = 3
_REQUIRED_RANK = 7
_MAX_VALID_ROTATION_ARCSEC = 3600.0


@dataclass(frozen=True, slots=True)
class Residual:
    """
    The discrepancy between a single control point's own known
    `target` and where the fitted transformation actually places its
    `source` -- `residual = target - transformed_source`, matching
    the sign convention already established by
    `topocore.geodesy.transform`'s own functions (never the reverse).
    """

    control_id: str
    dx: float
    dy: float
    dz: float
    magnitude: float


@dataclass(frozen=True, slots=True)
class HelmertFitResult:
    """
    Parameters
    ----------
    parameters
        The fitted `HelmertParameters`.
    residuals
        One `Residual` per control point, in the same order as given
        to `fit_helmert_3d()`.
    rms
        `sqrt(sum(magnitude**2) / n)` across all residuals -- the
        3D root-mean-square error of the fit, in the same linear
        units as the input coordinates.
    """

    parameters: HelmertParameters
    residuals: tuple[Residual, ...]
    rms: float


def apply_helmert_3d(
    point: tuple[float, float, float], parameters: HelmertParameters, *, apply_scale_to_z: bool = True
) -> tuple[float, float, float]:
    """
    Apply `parameters` directly to `point` -- plain Cartesian
    arithmetic, the position-vector formula documented at this
    module's own top (see there for the exact equations and why this
    is not the same as `CoordinateTransformer`'s own HELMERT path).

    `apply_scale_to_z` defaults to `True` -- the genuine 3D
    position-vector formula, matching this function's own historical
    behavior and every existing caller of it (unaffected by this
    parameter's addition). Set `False` specifically when `parameters`
    came from `fit_helmert_2d()`: confirmed directly this is required
    for consistency -- `HELMERT_2D`'s own model assumes `Z' = Z + tz`
    alone (no scale term at all, see `fit_helmert_2d()`'s own
    top-level docstring), so applying its result to REAL data (not
    just its own control points, whose Z range is typically small)
    with `apply_scale_to_z=True` would introduce a real, unbounded-
    with-Z discrepancy the fit's own residuals never saw or
    accounted for (confirmed directly: with Z=500 and a modest
    50 ppm scale, this introduces a 2.5 cm error in Z alone --
    growing linearly with |Z|, entirely invisible to
    `GeoreferencingResult.rms`, which only ever evaluated the
    control points themselves). `HelmertParameters` itself carries no
    "came from a 2D fit" information (a deliberate design choice --
    see this capability's own architecture decisions), so the
    caller -- which does know which strategy produced these
    parameters -- must state this explicitly; it is never inferred
    from the parameter values alone (e.g. `rx=ry=0` can also occur,
    if coincidentally, in a genuine 3D fit that should still scale
    Z).
    """
    xs, ys, zs = point
    s = parameters.scale * 1e-6
    rx = radians(parameters.rx / 3600.0)
    ry = radians(parameters.ry / 3600.0)
    rz = radians(parameters.rz / 3600.0)

    xt = xs + parameters.tx + s * xs - rz * ys + ry * zs
    yt = ys + parameters.ty + s * ys + rz * xs - rx * zs
    z_scale_term = s * zs if apply_scale_to_z else 0.0
    zt = zs + parameters.tz + z_scale_term - ry * xs + rx * ys
    return (xt, yt, zt)


def fit_helmert_3d(controls: Sequence[ControlPoint]) -> HelmertFitResult:
    """
    Estimate a 3D Helmert similarity transformation from `controls`
    via linear least-squares (position-vector convention, linearized
    for small rotations -- see this module's own top-level
    docstring).

    Raises
    ------
    InsufficientControlPointsError
        If fewer than 3 control points are given.
    DegenerateGeometryError
        If the control points, though numerous enough, do not
        provide 7 independent equations for the fit (e.g. collinear
        points) -- confirmed directly this is detected via the
        design matrix's own rank, never silently accepted.
    """
    if len(controls) < _MIN_CONTROL_POINTS:
        raise InsufficientControlPointsError(
            f"fit_helmert_3d() requires at least {_MIN_CONTROL_POINTS} control points "
            f"(a 3D Helmert fit has 7 unknowns), got {len(controls)}."
        )

    n = len(controls)
    design_matrix = np.zeros((3 * n, 7))
    observations = np.zeros(3 * n)

    for i, control in enumerate(controls):
        xs, ys, zs = control.source
        xt, yt, zt = control.target

        design_matrix[3 * i, :] = [1, 0, 0, xs, 0, zs, -ys]
        design_matrix[3 * i + 1, :] = [0, 1, 0, ys, -zs, 0, xs]
        design_matrix[3 * i + 2, :] = [0, 0, 1, zs, ys, -xs, 0]

        observations[3 * i : 3 * i + 3] = [xt - xs, yt - ys, zt - zs]

    solution, _sum_sq_residuals, rank, _singular_values = np.linalg.lstsq(design_matrix, observations, rcond=None)

    if rank < _REQUIRED_RANK:
        raise DegenerateGeometryError(
            f"fit_helmert_3d(): the {n} control points given do not provide {_REQUIRED_RANK} "
            f"independent equations (design matrix rank is {rank}) -- likely collinear or "
            "otherwise degenerately distributed. Provide control points with genuine 3D spread."
        )

    tx, ty, tz, s, rx, ry, rz = solution
    parameters = HelmertParameters(
        tx=float(tx),
        ty=float(ty),
        tz=float(tz),
        rx=degrees(rx) * 3600.0,
        ry=degrees(ry) * 3600.0,
        rz=degrees(rz) * 3600.0,
        scale=float(s) * 1e6,
    )

    if max(abs(parameters.rx), abs(parameters.ry), abs(parameters.rz)) > _MAX_VALID_ROTATION_ARCSEC:
        raise LinearizationInvalidError(
            f"fit_helmert_3d(): the fitted rotation (rx={parameters.rx:.1f}, ry={parameters.ry:.1f}, "
            f"rz={parameters.rz:.1f} arcsec) exceeds {_MAX_VALID_ROTATION_ARCSEC:.0f} arcsec (1 degree) -- "
            "beyond the range where this fit's own small-angle linearization remains valid. This "
            "typically means the control points cannot be related by a genuine small-rotation "
            "Helmert similarity transform at all (e.g. a mirror reflection between source and "
            "target, or badly mismatched correspondences), not merely an unusually large but "
            "still valid rotation."
        )

    residuals = []
    squared_magnitudes = []
    for control in controls:
        transformed_source = apply_helmert_3d(control.source, parameters)
        dx = control.target[0] - transformed_source[0]
        dy = control.target[1] - transformed_source[1]
        dz = control.target[2] - transformed_source[2]
        magnitude = float(np.sqrt(dx**2 + dy**2 + dz**2))
        residuals.append(Residual(control_id=control.id, dx=dx, dy=dy, dz=dz, magnitude=magnitude))
        squared_magnitudes.append(magnitude**2)

    rms = float(np.sqrt(np.mean(squared_magnitudes)))

    return HelmertFitResult(parameters=parameters, residuals=tuple(residuals), rms=rms)


__all__ = ["HelmertFitResult", "Residual", "apply_helmert_3d", "fit_helmert_3d"]
