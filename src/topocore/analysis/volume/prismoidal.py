"""
topocore.analysis.volume.prismoidal
====================================

Prismoidal volume computation using composite Simpson's 1/3 rule.

The calculation uses measured cross-sectional areas at uniformly
spaced stations.

For three sections:

    V = h / 3 * (A0 + 4*A1 + A2)

For multiple sections, the composite Simpson's 1/3 rule is applied.

Requirements
------------
- At least 3 sections.
- Odd number of sections.
- Strictly increasing stations.
- Uniform station spacing.
- Finite, non-negative areas.

The intermediate section areas are the measured values supplied by
the caller; they are not interpolated from the endpoint areas.

The class stores the sections provided at construction time and
exposes both ``compute()`` and the callable interface ``__call__()``.

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

from topocore.analysis.exceptions import VolumeError
from topocore.analysis.types import VolumeResult


class PrismoidalVolume:
    """Compute volume using the composite Simpson's 1/3 rule.

    Parameters
    ----------
    sections:
        Sequence of ``(station, area)`` pairs.

    Requirements
    ------------
    - At least three sections.
    - An odd number of sections.
    - Strictly increasing stations.
    - Uniform station spacing.
    - Finite, non-negative areas.

    The intermediate section areas supplied by the caller are used
    directly. They are not synthesized from endpoint areas.
    """

    __slots__ = ("_sections",)

    def __init__(
        self,
        sections: Sequence[tuple[float, float]],
    ) -> None:
        if len(sections) < 3:
            raise VolumeError("Prismoidal volume requires at least three sections.")

        if len(sections) % 2 == 0:
            raise VolumeError("Prismoidal volume requires an odd number of sections.")

        normalized = [(float(station), float(area)) for station, area in sections]

        for index, (station, area) in enumerate(normalized):
            if not math.isfinite(station):
                raise VolumeError(f"Section station at index {index} must be finite.")

            if not math.isfinite(area):
                raise VolumeError(f"Section area at index {index} must be finite.")

            if area < 0.0:
                raise VolumeError(f"Section area at index {index} cannot be negative.")

        spacings = [normalized[index + 1][0] - normalized[index][0] for index in range(len(normalized) - 1)]

        if any(spacing <= 0.0 for spacing in spacings):
            raise VolumeError("Section stations must be strictly increasing.")

        spacing = spacings[0]
        tolerance = max(1.0, abs(spacing)) * 1e-9

        if any(abs(current - spacing) > tolerance for current in spacings[1:]):
            raise VolumeError("Prismoidal sections must have uniformly spaced stations.")

        self._sections = normalized

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def section_count(self) -> int:
        """Number of cross sections."""
        return len(self._sections)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> VolumeResult:
        """Compute volume using composite Simpson's 1/3 rule."""
        areas = [area for _, area in self._sections]

        spacing = self._sections[1][0] - self._sections[0][0]

        volume = spacing / 3.0 * (areas[0] + areas[-1] + 4.0 * sum(areas[1:-1:2]) + 2.0 * sum(areas[2:-1:2]))

        return VolumeResult(
            cut_volume=volume,
            fill_volume=0.0,
            net_volume=volume,
            method="prismoidal",
            valid_cells=None,
            excluded_cells=None,
        )

    # ------------------------------------------------------------------
    # Segment volumes
    # ------------------------------------------------------------------

    def segment_volumes(self) -> list[float]:
        """Compute Simpson volumes for each prismoidal panel.

        Each panel consists of three consecutive sections:

            S0 ---- S1 ---- S2
            A0      A1      A2

        The returned values therefore correspond to:

            [S0,S1,S2], [S2,S3,S4], ...

        and their sum equals :meth:`compute`.
        """
        volumes: list[float] = []

        for index in range(0, len(self._sections) - 2, 2):
            station1, area1 = self._sections[index]
            station2, area2 = self._sections[index + 1]
            station3, area3 = self._sections[index + 2]

            spacing = station2 - station1

            # Validation was already performed in __init__.
            # Simpson's 1/3 rule over two equal intervals.
            volume = spacing / 3.0 * (area1 + 4.0 * area2 + area3)

            # station3 is intentionally part of the panel validation
            # and documents the three-section Simpson interval.
            _ = station3

            volumes.append(volume)

        return volumes

    # ------------------------------------------------------------------
    # Callable
    # ------------------------------------------------------------------

    def __call__(self) -> VolumeResult:
        """Compute the prismoidal volume."""
        return self.compute()


__all__ = [
    "PrismoidalVolume",
]
