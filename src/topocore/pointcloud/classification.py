"""
Point cloud classification definitions.

This module implements the ASPRS LAS classification codes.
Classification flags are intentionally implemented in a separate module.

References
----------
ASPRS LAS Specification 1.4 Revision 15
"""

from __future__ import annotations

from enum import IntEnum
from typing import final


@final
class PointClassification(IntEnum):
    """
    ASPRS LAS classification codes.
    """

    CREATED_NEVER_CLASSIFIED = 0
    UNCLASSIFIED = 1
    GROUND = 2
    LOW_VEGETATION = 3
    MEDIUM_VEGETATION = 4
    HIGH_VEGETATION = 5
    BUILDING = 6
    LOW_POINT = 7
    MODEL_KEY_POINT = 8
    WATER = 9
    RAIL = 10
    ROAD_SURFACE = 11
    OVERLAP = 12
    WIRE_GUARD = 13
    WIRE_CONDUCTOR = 14
    TRANSMISSION_TOWER = 15
    WIRE_STRUCTURE = 16
    BRIDGE_DECK = 17
    HIGH_NOISE = 18

    @property
    def code(self) -> int:
        """Returns the numeric classification code."""
        return int(self)

    @property
    def label(self) -> str:
        """Returns a human-readable label."""
        return self.name.replace("_", " ").title()

    @property
    def is_ground(self) -> bool:
        return self is PointClassification.GROUND

    @property
    def is_terrain(self) -> bool:
        return self is PointClassification.GROUND

    @property
    def is_building(self) -> bool:
        return self is PointClassification.BUILDING

    @property
    def is_water(self) -> bool:
        return self is PointClassification.WATER

    @property
    def is_noise(self) -> bool:
        return self in (
            PointClassification.LOW_POINT,
            PointClassification.HIGH_NOISE,
        )

    @property
    def is_vegetation(self) -> bool:
        return self in (
            PointClassification.LOW_VEGETATION,
            PointClassification.MEDIUM_VEGETATION,
            PointClassification.HIGH_VEGETATION,
        )

    @property
    def is_wire(self) -> bool:
        return self in (
            PointClassification.WIRE_GUARD,
            PointClassification.WIRE_CONDUCTOR,
            PointClassification.WIRE_STRUCTURE,
        )

    @classmethod
    def from_code(
        cls,
        code: int,
    ) -> PointClassification:
        """
        Creates a classification from its numeric code.
        """
        return cls(code)

    @classmethod
    def is_valid(
        cls,
        code: int,
    ) -> bool:
        """
        Returns True if the code is a valid ASPRS classification.
        """
        try:
            cls(code)
            return True
        except ValueError:
            return False


__all__ = [
    "PointClassification",
]
