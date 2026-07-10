from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ellipsoid:
    name: str
    semi_major_axis: float
    semi_minor_axis: float
    inverse_flattening: float
    is_semi_minor_computed: bool
