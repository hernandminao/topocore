"""
topocore.geodesy.georeferencing.single_control
=================================================

`fit_translation_only()`: the `TRANSLATION_ONLY` strategy --
`tx`/`ty`/`tz` estimated as the mean of `target - source` across all
controls (reducing to an exact value for the single-control-point
case); `rx`/`ry`/`rz` and `scale` are never estimated, always
assumed `0.0`.

This is a genuinely simpler fit than `fit_helmert_3d()`/2D Helmert:
translation has no rank-deficiency case at all -- each axis (X, Y, Z)
is fully independent (`tx` only ever appears in the X equation, `ty`
only in Y, `tz` only in Z, with no cross-coupling the way rotation/
scale create), so this works identically for any `n >= 1`, never
raising `DegenerateGeometryError`.

Returns `HelmertFitResult` (the same type `fit_helmert_3d()` uses),
whose own `rms: float` field is always a genuine number -- for a
single control point this is trivially `0.0` (3 unknowns, 3
equations, no redundancy at all). Deciding whether this trivial
value should instead be reported as "not applicable"
(`GeoreferencingResult.rms: float | None`, `None` specifically for
that single-point case) is `fit_georeferencing()`'s own
responsibility once it assembles the final result -- not this
function's, which always reports a real, meaningful number for
whatever `controls` it was actually given, including `n > 1` (a
translation-only fallback from a degenerate higher-rank attempt,
where genuine redundancy -- and a genuinely informative RMS -- does
exist).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from topocore.geodesy.georeferencing.control_point import ControlPoint
from topocore.geodesy.georeferencing.exceptions import InsufficientControlPointsError
from topocore.geodesy.georeferencing.helmert_fit import (
    HelmertFitResult,
    Residual,
    apply_helmert_3d,
)
from topocore.geodesy.helmert import HelmertParameters


def fit_translation_only(controls: Sequence[ControlPoint]) -> HelmertFitResult:
    """
    Estimate a translation-only transformation (`rx=ry=rz=0.0`,
    `scale=0.0` ppm, always assumed, never estimated) from `controls`.

    `tx`/`ty`/`tz` are each the mean of `target - source` across all
    controls, verified directly this reduces to an exact value for a
    single control point (the common case this strategy exists for),
    and genuinely averages -- with real, nonzero residuals when there
    is real inconsistency -- for more than one.

    Raises
    ------
    InsufficientControlPointsError
        If `controls` is empty.
    """
    if len(controls) < 1:
        raise InsufficientControlPointsError("fit_translation_only() requires at least 1 control point, got 0.")

    deltas = np.array(
        [
            [
                control.target[0] - control.source[0],
                control.target[1] - control.source[1],
                control.target[2] - control.source[2],
            ]
            for control in controls
        ]
    )
    mean_delta = deltas.mean(axis=0)

    parameters = HelmertParameters(
        tx=float(mean_delta[0]),
        ty=float(mean_delta[1]),
        tz=float(mean_delta[2]),
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


__all__ = ["fit_translation_only"]
