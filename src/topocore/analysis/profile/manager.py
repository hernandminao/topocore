"""
topocore.analysis.profile.manager
==================================

Unified profile analysis facade.

Provides a single entry point for generating terrain profiles
using any supported method.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from topocore.analysis.config import (
    DEFAULT_ANALYSIS_CONFIG,
    ProfileConfig,
)
from topocore.analysis.exceptions import ProfileError
from topocore.analysis.protocols import TerrainSurface
from topocore.analysis.types import ProfileResult

from .cross_section import CrossSectionProfile
from .longitudinal import LongitudinalProfile
from .multi_profile import MultiProfile
from .transversal import TransversalProfile

# ============================================================================
# Methods
# ============================================================================


class ProfileMethod(StrEnum):
    """Supported profile generation methods."""

    LONGITUDINAL = "longitudinal"
    TRANSVERSAL = "transversal"
    CROSS_SECTION = "cross_section"
    MULTI = "multi"


# ============================================================================
# Manager
# ============================================================================


class ProfileAnalysis:
    """
    Unified profile analysis manager.

    Provides a high-level interface for all profile generation
    algorithms supported by TopoCore.

    Parameters
    ----------
    config
        Profile configuration.
    method
        Default generation method.
    """

    __slots__ = (
        "_config",
        "_interval",
        "_method",
        "_width",
    )

    def __init__(
        self,
        *,
        config: ProfileConfig | None = None,
        method: str | None = None,
    ) -> None:

        self._config = config or DEFAULT_ANALYSIS_CONFIG.profile

        resolved_method = method or self._config.default_method

        if resolved_method not in ProfileMethod:
            raise ProfileError(
                f"Unsupported profile method '{resolved_method}'. Available: {[m.value for m in ProfileMethod]}"
            )

        self._method = ProfileMethod(resolved_method)

        self._interval = float(self._config.default_interval)

        self._width = float(self._config.default_width)

        if self._interval <= 0:
            raise ProfileError("Profile interval must be positive.")

        if self._width <= 0:
            raise ProfileError("Profile width must be positive.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Current profile method."""

        return self._method.value

    @property
    def config(self) -> ProfileConfig:
        """Current profile configuration."""

        return self._config

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def longitudinal(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        surface: TerrainSurface,
        *,
        interval: float | None = None,
    ) -> ProfileResult:
        """
        Generate longitudinal profile.
        """

        generator = LongitudinalProfile(interval=(self._interval if interval is None else interval))

        return generator.generate(
            origin,
            target,
            surface,
        )

    def transversal(
        self,
        axis_origin: tuple[float, float],
        axis_target: tuple[float, float],
        station: float,
        surface: TerrainSurface,
        *,
        interval: float | None = None,
        width: float | None = None,
    ) -> ProfileResult:
        """
        Generate transversal profile.
        """

        generator = TransversalProfile(
            interval=(self._interval if interval is None else interval),
            width=(self._width if width is None else width),
        )

        return generator.generate(
            axis_origin,
            axis_target,
            station,
            surface,
        )

    def cross_section(
        self,
        axis: list[tuple[float, float]],
        surface: TerrainSurface,
        *,
        interval: float | None = None,
        width: float | None = None,
    ) -> list[ProfileResult]:
        """
        Generate cross sections along an alignment.
        """

        generator = CrossSectionProfile(
            interval=(self._interval if interval is None else interval),
            width=(self._width if width is None else width),
        )

        return generator.generate(
            axis,
            surface,
        )

    def multi(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        surface: TerrainSurface,
        *,
        offsets: list[float] | None = None,
        interval: float | None = None,
    ) -> list[ProfileResult]:
        """
        Generate multiple parallel profiles.
        """

        generator = MultiProfile(
            interval=(self._interval if interval is None else interval),
            offsets=offsets,
        )

        return generator.generate(
            origin,
            target,
            surface,
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def compute(
        self,
        *args: Any,
        method: str | None = None,
        **kwargs: Any,
    ) -> ProfileResult | list[ProfileResult]:
        """
        Generate profile using selected method.
        """

        try:
            selected = ProfileMethod(method or self._method.value)
        except ValueError as exc:
            raise ProfileError(
                f"Unsupported profile method '{method}'. Available: {[m.value for m in ProfileMethod]}"
            ) from exc

        if selected is ProfileMethod.LONGITUDINAL:
            return self.longitudinal(
                *args,
                **kwargs,
            )

        if selected is ProfileMethod.TRANSVERSAL:
            return self.transversal(
                *args,
                **kwargs,
            )

        if selected is ProfileMethod.CROSS_SECTION:
            return self.cross_section(
                *args,
                **kwargs,
            )

        if selected is ProfileMethod.MULTI:
            return self.multi(
                *args,
                **kwargs,
            )

        raise ProfileError(f"Unsupported profile method '{selected}'.")

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ProfileResult | list[ProfileResult]:
        """
        Callable interface.
        """

        return self.compute(
            *args,
            **kwargs,
        )


__all__ = [
    "ProfileAnalysis",
    "ProfileMethod",
]
