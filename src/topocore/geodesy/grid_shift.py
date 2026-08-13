"""
topocore.geodesy.grid_shift
===============================

Describes a grid-shift resource (NTv2 .gsb, GeoTIFF grid) without
loading it, using PROJ, or caching anything -- pure data, no
behavior. Loading and applying a grid shift belongs to a later PR
(PR18A.3+), once there's a real consumer that needs it.

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
from pathlib import Path

from topocore.geodesy.exceptions import ValidationError


class GridShiftFormat(StrEnum):
    """
    Grid-shift file formats supported as of PR18A.2 -- not an
    exhaustive list of every format PROJ itself can consume (e.g.
    GTX). New formats are added when a real consumer needs them,
    same rule as everywhere else in this module.
    """

    NTV2 = "ntv2"
    GEOTIFF = "geotiff"


@dataclass(frozen=True, slots=True)
class GridShift:
    """
    Describes a grid-shift resource. Never loads `path`, never opens
    it, never validates it exists -- consistent with `Datum`/
    `Ellipsoid`/`ProjectionInfo`, none of which validate their own
    fields either; validation lives in `validation.py`, applied by
    whatever eventually consumes this.

    Parameters
    ----------
    path
        Location of the grid file.
    format
        See `GridShiftFormat`.
    description
        Human-readable description (e.g. "Colombia MAGNA-SIRGAS shift grid").
    area_of_use
        Textual description of the grid's valid coverage area.
    accuracy
        Estimated accuracy in meters, if known. Must not be negative.
    """

    path: Path
    format: GridShiftFormat
    description: str | None = None
    area_of_use: str | None = None
    accuracy: float | None = None

    def __post_init__(self) -> None:
        if self.accuracy is not None and self.accuracy < 0:
            raise ValidationError(f"GridShift.accuracy cannot be negative, got {self.accuracy}.")


__all__ = ["GridShift", "GridShiftFormat"]
