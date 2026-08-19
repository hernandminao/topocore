"""
Shared test helper for topocore.analysis.profile -- PR19.

TerrainSurface (the Protocol every profile generator expects) requires
BOTH interpolate(x, y) and contains(x, y). topocore.terrain.linear.
LinearInterpolator only implements interpolate() -- it's a low-level
TIN interpolator, not itself meant to BE a TerrainSurface. It happened
to work at runtime throughout this session's profile tests because
none of the exercised code paths ever call contains() (only
interpolate_surface() in _shared/surface.py is used, which never
touches contains()) -- but mypy correctly flags the structural
mismatch. This tiny wrapper is the correct, protocol-conforming test
double: delegates interpolate() to the real interpolator and contains()
to the underlying TIN's own contains(), which already exists.
"""

from __future__ import annotations

from topocore.terrain.linear import LinearInterpolator
from topocore.terrain.tin import TIN


class SurfaceAdapter:
    """Wraps a LinearInterpolator to fully satisfy TerrainSurface."""

    __slots__ = ("_interpolator", "_tin")

    def __init__(self, tin: TIN) -> None:
        self._tin = tin
        self._interpolator = LinearInterpolator(tin)

    def interpolate(self, x: float, y: float) -> float:
        return self._interpolator.interpolate(x, y)

    def contains(self, x: float, y: float) -> bool:
        return self._tin.contains(x, y)
