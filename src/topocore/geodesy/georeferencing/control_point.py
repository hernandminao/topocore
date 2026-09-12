"""
topocore.geodesy.georeferencing.control_point
================================================

`ControlPoint`: a single known correspondence between a local/
arbitrary source coordinate and its known target coordinate --
distinct from `topocore.survey.models.SurveyPoint`, which represents
a single observed position, not a correspondence between two.

Deliberately independent of any CRS/PROJ concept: `source` and
`target` are plain Cartesian (X, Y, Z) tuples, arbitrary units --
this module has no opinion on what coordinate system either one is
actually expressed in. That association (a `target_crs`) belongs to
a later phase of this same capability, not this pure-geometry core.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from topocore.geodesy.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class ControlPoint:
    """
    Parameters
    ----------
    id
        Identifies this control point (e.g. matching a
        `SurveyPoint.id` from the same survey) -- carried through
        purely for traceability, never interpreted.
    source
        The local/arbitrary (X, Y, Z) coordinate as actually
        observed/surveyed.
    target
        The known (X, Y, Z) coordinate this same physical point is
        supposed to correspond to.
    """

    id: str
    source: tuple[float, float, float]
    target: tuple[float, float, float]

    def __post_init__(self) -> None:
        for label, value in (("source", self.source), ("target", self.target)):
            if len(value) != 3:
                raise ValidationError(
                    f"ControlPoint.{label} must have exactly 3 components (X, Y, Z), got {len(value)}."
                )
            if not all(isfinite(component) for component in value):
                raise ValidationError(f"ControlPoint.{label} contains a NaN or infinite value: {value!r}.")


__all__ = ["ControlPoint"]
