from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DXFTolerance:
    z_planarity: float = 1e-6
    coordinate: float = 1e-9

    def __post_init__(self) -> None:
        if self.z_planarity < 0.0:
            raise ValueError(f"z_planarity must be non-negative; got {self.z_planarity}.")
        if self.coordinate < 0.0:
            raise ValueError(f"coordinate must be non-negative; got {self.coordinate}.")


__all__ = ["DXFTolerance"]
