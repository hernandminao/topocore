"""
topocore.analysis.volume.prismoidal
====================================

Prismoidal volume computation.

Applies the prismoidal formula along a series of parallel cross
sections.

Formula:

    V = L / 6 * (A1 + 4*Am + A2)

where:

- L  = distance between sections
- A1 = first section area
- A2 = second section area
- Am = intermediate section area

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Sequence

from topocore.analysis.exceptions import VolumeError
from topocore.analysis.types import VolumeResult


class PrismoidalVolume:
    """
    Computes volumes using the prismoidal formula.

    Parameters
    ----------
    sections
        Ordered sequence of ``(station, area)`` pairs.
    """

    __slots__ = ("_sections",)

    def __init__(
        self,
        sections: Sequence[tuple[float, float]],
    ) -> None:

        if len(sections) < 2:
            raise VolumeError("Prismoidal formula requires at least 2 cross sections.")

        normalized = list(sections)

        for i in range(1, len(normalized)):
            previous_station = normalized[i - 1][0]
            current_station = normalized[i][0]

            if current_station <= previous_station:
                raise VolumeError("Cross sections must be ordered by increasing station.")

        for station, area in normalized:
            if area < 0:
                raise VolumeError(f"Section area cannot be negative (station={station}).")

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
        """
        Compute total prismoidal volume.

        Returns
        -------
        VolumeResult
            Total computed volume.
        """

        total_volume = 0.0

        for segment_volume in self.segment_volumes():
            total_volume += segment_volume

        return VolumeResult(
            cut_volume=total_volume,
            fill_volume=0.0,
            net_volume=total_volume,
            method="prismoidal",
        )

    def segment_volumes(self) -> list[float]:
        """
        Compute individual segment volumes.

        Returns
        -------
        list[float]
            Volume of each interval.
        """

        volumes: list[float] = []

        for i in range(len(self._sections) - 1):
            station1, area1 = self._sections[i]
            station2, area2 = self._sections[i + 1]

            length = station2 - station1

            #
            # Without an explicit middle section, use
            # the average area approximation.
            #
            middle_area = (area1 + area2) / 2.0

            volume = length / 6.0 * (area1 + 4.0 * middle_area + area2)

            volumes.append(volume)

        return volumes

    # ------------------------------------------------------------------
    # Callable
    # ------------------------------------------------------------------

    def __call__(self) -> VolumeResult:
        return self.compute()


__all__ = [
    "PrismoidalVolume",
]
