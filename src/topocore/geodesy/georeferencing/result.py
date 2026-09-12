"""
topocore.geodesy.georeferencing.result
=========================================

`GeoreferencingResult`: the outcome of `fit_georeferencing()` --
which strategy was actually used, the fitted `HelmertParameters`,
per-control residuals, and (when applicable) the overall RMS.

Kept in its own module, separate from `helmert_fit.py`'s own
`HelmertFitResult`: that type is specifically the output of
`fit_helmert_3d()` alone, while `GeoreferencingResult` represents a
choice made among 3 possible strategies (`TRANSLATION_ONLY`,
`HELMERT_2D`, `HELMERT_3D`), never assumed to be 3D.

`strategy`/`warning` carry the epistemic distinction between
"measured" and "assumed" parameters -- `HelmertParameters` itself
never does, and is not extended to do so (confirmed during this
capability's own audit: its own `scale=0.0`/`rx=ry=rz=0.0` defaults
already mean "no scale/rotation" for whatever the class represents,
whether that's a genuinely known translation-only EPSG-registered
transform or a georeferencing result where those same zeros were
never actually measured -- `HelmertParameters` cannot and should not
try to distinguish the two; that distinction belongs here).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from topocore.geodesy.georeferencing.helmert_fit import Residual
from topocore.geodesy.helmert import HelmertParameters


class GeoreferencingStrategy(StrEnum):
    """Which of the 3 possible fits `fit_georeferencing()` actually used."""

    TRANSLATION_ONLY = "translation_only"
    HELMERT_2D = "helmert_2d"
    HELMERT_3D = "helmert_3d"


@dataclass(frozen=True, slots=True)
class GeoreferencingResult:
    """
    Parameters
    ----------
    strategy
        Which fit was actually used.
    parameters
        The fitted `HelmertParameters`. For `TRANSLATION_ONLY`,
        `rx`/`ry`/`rz` are `0.0` and `scale` is `0.0` (ppm) --
        never actually estimated, only assumed; for `HELMERT_2D`,
        `rx`/`ry` are likewise assumed `0.0`. See this module's own
        docstring for why this distinction lives here, not on
        `HelmertParameters` itself.
    residuals
        One `Residual` per control point, in the same order given to
        `fit_georeferencing()`.
    rms
        `None` when there is no genuine redundancy to measure --
        confirmed directly this is exactly the single-control-point
        `TRANSLATION_ONLY` case (3 unknowns, 3 equations, no room for
        inconsistency at all, so a numerically-zero RMS would be
        redundant, not informative). A real `float` in every other
        case, including `TRANSLATION_ONLY` fitted from more than one
        control point (e.g. as a fallback from a degenerate 2D/3D
        attempt), where genuine redundancy -- and therefore a
        meaningful RMS -- exists.
    control_count
        `len(controls)` given to `fit_georeferencing()`.
    warning
        `None` when `strategy` is fully determined by the data
        (`HELMERT_3D` with a full-rank fit). A human-readable
        explanation of exactly which parameters were assumed
        (never just "some parameters were assumed") whenever
        `strategy` is `TRANSLATION_ONLY` or `HELMERT_2D`.
    """

    strategy: GeoreferencingStrategy
    parameters: HelmertParameters
    residuals: tuple[Residual, ...]
    rms: float | None
    control_count: int
    warning: str | None


__all__ = ["GeoreferencingResult", "GeoreferencingStrategy"]
