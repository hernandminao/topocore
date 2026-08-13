"""
topocore.geodesy.local_crs
==============================

Represents a project-local (engineering/construction site)
coordinate system: an origin, rotation, and scale relative to a real
`CRS`. Never inherits from `CRS`, never replaces it -- references
it. Pure data; transforming coordinates through a `LocalCRS` belongs
to a later PR, once `CoordinateOperation` exists.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class LocalCRS:
    """
    Parameters
    ----------
    name
        Project or site name (e.g. "Proyecto Norte"). Cannot be empty.
    base_crs
        The real CRS this local system is defined relative to.
    origin_x, origin_y
        Origin of the local system, in `base_crs` coordinates. No
        default -- every local CRS has a distinct, explicit origin.
    rotation
        Rotation, in decimal degrees, counter-clockwise, relative to
        the positive X axis. Default 0.0 -- many project-local
        systems genuinely have no rotation; this is an identity
        default, not a guessed algorithm choice. This convention is
        documented now purely to freeze intent; it has no behavior
        to validate against yet (no transform exists) and should be
        re-checked against whatever rotation matrix a future PR
        actually implements.
    scale
        Scale factor. Default 1.0 (no scaling). Must be positive --
        a zero or negative scale collapses or mirrors every
        coordinate, which is never a meaningful engineering CRS.
    false_easting, false_northing
        Offsets applied to local coordinates. Default 0.0 (none).
    """

    name: str
    base_crs: CRS
    origin_x: float
    origin_y: float
    rotation: float = 0.0
    scale: float = 1.0
    false_easting: float = 0.0
    false_northing: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("LocalCRS.name cannot be empty.")
        if self.scale <= 0:
            raise ValidationError(f"LocalCRS.scale must be positive, got {self.scale}.")


__all__ = ["LocalCRS"]
