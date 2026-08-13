"""
topocore.geodesy.accuracy
=============================

`TransformationAccuracy`: the precision of an actually-executed (or
described) coordinate transformation -- distinct from
`ProjectionInfo.accuracy`, which describes the static accuracy a
CRS's own registered coordinate operation carries (from the EPSG
registry), not the accuracy of a specific transformation a caller
performs. Two different domains; see PR18A design discussion.

Pure value object -- computing or estimating an accuracy belongs to
whatever produces a `CoordinateOperation`, not to this class.

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
class TransformationAccuracy:
    """
    Parameters
    ----------
    value
        Numeric accuracy value. Must not be negative.
    unit
        Unit `value` is expressed in. Default "m" (meters) -- the
        overwhelming majority of geodetic accuracy figures are
        reported in meters; a different unit is explicit, not implied.
    source
        Where the figure comes from (e.g. "EPSG registry", "NTv2 grid metadata",
        "user-supplied"). `None` if unknown.
    estimated
        True if `value` is an estimate/approximation rather than a
        figure from an authoritative source. Default False.
    """

    value: float
    unit: str = "m"
    source: str | None = None
    estimated: bool = False

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValidationError(f"TransformationAccuracy.value cannot be negative, got {self.value}.")


__all__ = ["TransformationAccuracy"]
