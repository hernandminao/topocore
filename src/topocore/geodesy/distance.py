from __future__ import annotations
import numpy as np

def distance2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate 2D Euclidean distance."""
    return float(np.hypot(x2 - x1, y2 - y1))

def distance3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """Calculate 3D Euclidean distance."""
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))

__all__ = ["distance2d", "distance3d"]
