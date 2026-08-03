"""
topocore.analysis.volume.average_end_area
==========================================

Average end-area volume computation.

Computes volume between consecutive cross sections using:

    V = L * (A1 + A2) / 2

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

Section = tuple[float, float]


class AverageEndAreaVolume:
    """
    Computes volumes using the average end-area method.

    Parameters
    ----------
    sections
        Ordered sequence of ``(station, area)`` pairs.
    """

    __slots__ = ("_sections",)

    def __init__(
        self,
        sections: Sequence[Section],
    ) -> None:

        if len(sections) < 2:
            raise VolumeError("Average end-area method requires at least 2 sections.")

        normalized: list[Section] = []

        for station, area in sections:
            if not math.isfinite(station):
                raise VolumeError("Section station must be finite.")

            if not math.isfinite(area):
                raise VolumeError("Section area must be finite.")

            if area < 0:
                raise VolumeError("Section areas cannot be negative.")

            normalized.append(
                (
                    float(station),
                    float(area),
                )
            )

        for i in range(1, len(normalized)):
            if normalized[i][0] <= normalized[i - 1][0]:
                raise VolumeError("Sections must be ordered by increasing station.")

        self._sections = normalized

    @property
    def section_count(self) -> int:
        """
        Number of cross sections.
        """
        return len(self._sections)

    def compute(self) -> VolumeResult:
        """
        Compute total volume.

        Returns
        -------
        VolumeResult
            Computed volume.
        """

        total = sum(
            self._segment_volume(
                self._sections[i],
                self._sections[i + 1],
            )
            for i in range(len(self._sections) - 1)
        )

        return VolumeResult(
            cut_volume=float(total),
            fill_volume=0.0,
            net_volume=float(total),
            method="average_end_area",
        )

    def segment_volumes(self) -> list[float]:
        """
        Compute individual segment volumes.
        """

        return [
            self._segment_volume(
                self._sections[i],
                self._sections[i + 1],
            )
            for i in range(len(self._sections) - 1)
        ]

    @staticmethod
    def _segment_volume(
        first: Section,
        second: Section,
    ) -> float:
        """
        Compute volume between two sections.
        """

        station1, area1 = first
        station2, area2 = second

        distance = station2 - station1

        return distance * ((area1 + area2) / 2.0)

    def __call__(self) -> VolumeResult:
        return self.compute()


__all__ = [
    "AverageEndAreaVolume",
]
