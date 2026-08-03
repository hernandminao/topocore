"""
topocore.analysis.visibility.manager
=====================================

Unified visibility analysis facade.

Provides a single entry point for visibility computations using
any supported method.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

from topocore.analysis.config import (
    DEFAULT_ANALYSIS_CONFIG,
    VisibilityConfig,
)
from topocore.analysis.exceptions import VisibilityError
from topocore.analysis.protocols import TriangulatedSurface
from topocore.analysis.types import (
    IntervisibilityResult,
    LOSResult,
    ViewshedResult,
)

from .intervisibility import Intervisibility
from .los import LineOfSight
from .viewshed import Viewshed

# ============================================================================
# Types
# ============================================================================

VisibilityResult: TypeAlias = LOSResult | ViewshedResult | IntervisibilityResult

VisibilityMethodType = Callable[..., VisibilityResult]


# ============================================================================
# Methods
# ============================================================================


class VisibilityMethod:
    """
    Supported visibility methods.
    """

    LINE_OF_SIGHT = "los"
    VIEWSHED = "viewshed"
    INTERVISIBILITY = "intervisibility"


_VALID_METHODS = {
    VisibilityMethod.LINE_OF_SIGHT,
    VisibilityMethod.VIEWSHED,
    VisibilityMethod.INTERVISIBILITY,
}


# ============================================================================
# Manager
# ============================================================================


class VisibilityAnalysis:
    """
    Unified visibility analysis manager.

    Parameters
    ----------
    config
        Visibility-specific configuration.
    """

    __slots__ = (
        "_config",
        "_method",
        "_observer_height",
        "_target_height",
        "_dispatch",
    )

    def __init__(
        self,
        *,
        config: VisibilityConfig | None = None,
        method: str | None = None,
    ) -> None:

        self._config = config or DEFAULT_ANALYSIS_CONFIG.visibility

        self._method = method or self._config.default_method

        if self._method not in _VALID_METHODS:
            raise VisibilityError(f"Unknown visibility method: '{self._method}'.")

        self._observer_height = self._config.observer_height

        self._target_height = self._config.target_height

        self._dispatch: dict[
            str,
            VisibilityMethodType,
        ] = {
            VisibilityMethod.LINE_OF_SIGHT: self.line_of_sight,
            VisibilityMethod.VIEWSHED: self.viewshed,
            VisibilityMethod.INTERVISIBILITY: self.intervisibility,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Configured visibility method."""
        return self._method

    @property
    def config(self) -> VisibilityConfig:
        """Visibility configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Algorithms
    # ------------------------------------------------------------------

    def line_of_sight(
        self,
        observer: tuple[float, float],
        target: tuple[float, float],
        tin: TriangulatedSurface,
        *,
        observer_height: float | None = None,
        target_height: float | None = None,
    ) -> LOSResult:
        """
        Compute line-of-sight visibility.
        """

        oh = observer_height if observer_height is not None else self._observer_height

        th = target_height if target_height is not None else self._target_height

        return LineOfSight(
            observer_height=oh,
            target_height=th,
            earth_curvature=(self._config.earth_curvature_correction),
        ).compute(
            observer,
            target,
            tin,
        )

    def viewshed(
        self,
        observer: tuple[float, float],
        tin: TriangulatedSurface,
        *,
        observer_height: float | None = None,
        resolution: float = 5.0,
        max_distance: float = 0.0,
    ) -> ViewshedResult:
        """
        Compute terrain viewshed.
        """

        oh = observer_height if observer_height is not None else self._observer_height

        return Viewshed(
            observer_height=oh,
            resolution=resolution,
            max_distance=max_distance,
        ).compute(
            observer,
            tin,
        )

    def intervisibility(
        self,
        points: list[tuple[float, float]],
        tin: TriangulatedSurface,
        *,
        observer_height: float | None = None,
    ) -> IntervisibilityResult:
        """
        Compute pairwise point visibility.
        """

        oh = observer_height if observer_height is not None else self._observer_height

        return Intervisibility(
            observer_height=oh,
            earth_curvature=(self._config.earth_curvature_correction),
        ).compute(
            points,
            tin,
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def compute(
        self,
        *args: Any,
        method: str | None = None,
        **kwargs: Any,
    ) -> VisibilityResult:
        """
        Execute configured visibility analysis.
        """

        target = method or self._method

        try:
            operation = self._dispatch[target]
        except KeyError as exc:
            raise VisibilityError(f"Unknown visibility method: '{target}'.") from exc

        return operation(
            *args,
            **kwargs,
        )

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> VisibilityResult:
        """
        Execute configured analysis.
        """

        return self.compute(
            *args,
            **kwargs,
        )


__all__ = [
    "VisibilityAnalysis",
    "VisibilityMethod",
    "VisibilityResult",
]
