"""
topocore.io.ascii.header
========================

ASCII header detection utilities.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

HEADER_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "x": (
        "x",
        "east",
        "easting",
        "longitude",
        "lon",
    ),
    "y": (
        "y",
        "north",
        "northing",
        "latitude",
        "lat",
    ),
    "z": (
        "z",
        "elevation",
        "elev",
        "height",
        "rl",
        "level",
    ),
    "id": (
        "id",
        "pointid",
        "point_id",
        "station",
        "name",
    ),
    "description": (
        "description",
        "desc",
        "remarks",
        "comment",
        "code",
        "feature",
    ),
    "classification": (
        "classification",
        "class",
    ),
    "intensity": ("intensity",),
    "red": (
        "red",
        "r",
    ),
    "green": (
        "green",
        "g",
    ),
    "blue": (
        "blue",
        "b",
    ),
}


@dataclass(slots=True, frozen=True)
class HeaderDetectionResult:
    """
    Result of header detection.
    """

    has_header: bool

    columns: dict[str, int]

    @classmethod
    def detect(
        cls,
        fields: list[str],
    ) -> HeaderDetectionResult:

        columns: dict[str, int] = {}

        for index, value in enumerate(fields):
            normalized = value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

            for name, aliases in HEADER_ALIASES.items():
                if normalized in aliases:
                    columns[name] = index
                    break

        return HeaderDetectionResult(
            has_header={"x", "y", "z"}.issubset(columns),
            columns=columns,
        )
