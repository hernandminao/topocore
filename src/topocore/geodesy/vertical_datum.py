"""
topocore.geodesy.vertical_datum
===================================

Represents a vertical datum (e.g. EGM96, EGM2008, NAVD88) -- pure
data, distinct from `Datum` (horizontal/geometric). Never mixed with
`CRS`. Computing ellipsoidal-to-orthometric height shifts belongs to
a later PR, once there's a real geoid model consumer.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.geodesy.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class VerticalDatum:
    """
    Parameters
    ----------
    name
        Vertical datum name (e.g. "EGM96", "NAVD88"). Cannot be empty.
    authority
        Defining authority (e.g. "NGA", "NOAA"), if known.
    geoid_model
        Name of the associated geoid model, if any.
    remarks
        Free-text remarks.
    """

    name: str
    authority: str | None = None
    geoid_model: str | None = None
    remarks: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("VerticalDatum.name cannot be empty.")


__all__ = ["VerticalDatum"]
