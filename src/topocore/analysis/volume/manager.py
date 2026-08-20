"""
topocore.analysis.volume.manager
=================================

Unified volume analysis facade.

Provides a single entry point for computing volumes using any
supported method.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any

from topocore.analysis.config import (
    DEFAULT_ANALYSIS_CONFIG,
    VolumeConfig,
)
from topocore.analysis.exceptions import VolumeError
from topocore.analysis.types import (
    VolumeMethod,
    VolumeResult,
)

from .average_end_area import AverageEndAreaVolume
from .cut_fill import CutFillVolume
from .grid_volume import GridVolume
from .prismoidal import PrismoidalVolume
from .tin_volume import TINVolume


class VolumeAnalysis:
    """
    Unified volume analysis manager.
    """

    __slots__ = (
        "_config",
        "_method",
    )

    def __init__(
        self,
        *,
        config: VolumeConfig | None = None,
        method: str | VolumeMethod | None = None,
    ) -> None:

        self._config = config or DEFAULT_ANALYSIS_CONFIG.volume

        resolved = method or self._config.default_method

        resolved_value = str(resolved)

        valid_methods = {item.value for item in VolumeMethod}

        if resolved_value not in valid_methods:
            raise VolumeError(f"Unsupported volume method: '{resolved_value}'. Choose from: {sorted(valid_methods)}.")

        self._method = resolved_value

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Current volume method."""
        return self._method

    @property
    def config(self) -> VolumeConfig:
        """Current configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Individual methods
    # ------------------------------------------------------------------

    def cut_fill(
        self,
        existing: Any,
        proposed: Any,
        cell_area: float,
    ) -> VolumeResult:
        """
        Compute cut/fill volume between two grids.
        """

        return CutFillVolume(cell_area).compute(
            existing,
            proposed,
        )

    def prismoidal(
        self,
        sections: list[tuple[float, float]],
    ) -> VolumeResult:
        """
        Compute prismoidal volume using the composite Simpson's 1/3
        rule.

        Requires an odd number of sections (at least 3), uniformly
        spaced, with genuinely measured intermediate areas -- not
        synthesized from the endpoints. See
        ``topocore.analysis.volume.prismoidal`` for the full
        contract.

        .. note::
            The PR19 finding that this method used to be
            mathematically equivalent to ``average_end_area()`` (the
            middle-section area was approximated as the average of
            the endpoints, defeating Simpson's rule) has since been
            resolved: ``PrismoidalVolume`` now requires real,
            measured intermediate sections and gives genuinely
            different (more accurate for curved cross-section
            variation) results. Confirmed directly against an exact
            analytic integral of a quadratic area profile.
        """

        return PrismoidalVolume(sections).compute()

    def average_end_area(
        self,
        sections: list[tuple[float, float]],
    ) -> VolumeResult:
        """
        Compute average end-area volume.
        """

        return AverageEndAreaVolume(sections).compute()

    def tin_volume(
        self,
        tin: Any,
        datum: float = 0.0,
    ) -> VolumeResult:
        """
        Compute volume under TIN.
        """

        return TINVolume(datum).compute(tin)

    def grid_volume(
        self,
        existing: Any,
        proposed: Any,
        resolution: float,
    ) -> VolumeResult:
        """
        Compute volume between grids.
        """

        return GridVolume(resolution).compute(
            existing,
            proposed,
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def compute(
        self,
        *args: Any,
        method: str | VolumeMethod | None = None,
        **kwargs: Any,
    ) -> VolumeResult:
        """
        Compute volume using selected method.
        """

        target = str(method or self._method)

        if target == VolumeMethod.CUT_FILL.value:
            return self.cut_fill(
                *args,
                **kwargs,
            )

        if target == VolumeMethod.PRISMOIDAL.value:
            return self.prismoidal(
                *args,
                **kwargs,
            )

        if target == VolumeMethod.AVERAGE_END_AREA.value:
            return self.average_end_area(
                *args,
                **kwargs,
            )

        if target == VolumeMethod.TIN_VOLUME.value:
            return self.tin_volume(
                *args,
                **kwargs,
            )

        if target == VolumeMethod.GRID_VOLUME.value:
            return self.grid_volume(
                *args,
                **kwargs,
            )

        raise VolumeError(f"Unknown volume method: '{target}'.")

    # ------------------------------------------------------------------
    # Callable
    # ------------------------------------------------------------------

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> VolumeResult:

        return self.compute(
            *args,
            **kwargs,
        )


__all__ = [
    "VolumeAnalysis",
]
