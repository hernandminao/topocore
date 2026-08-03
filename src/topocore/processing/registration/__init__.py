"""
topocore.processing.registration
================================

Point cloud registration.

This package provides various methods for registering (aligning)
point clouds:

- Point-to-Point ICP: Classic ICP using point correspondences
- Point-to-Plane ICP: ICP using tangent plane distances
- GICP: Generalized ICP (future)
- Colored ICP: ICP with color constraints (future)
- FGR: Fast Global Registration (future)
- RANSAC: Feature-based registration (future)

Registration is essential for:
- Combining multiple LiDAR scans
- Aligning drone and GNSS data
- Multi-temporal analysis
- Deformation monitoring
- Mining and earthworks

Public API
----------
- Transformation: 3D rigid transformation
- RegistrationResult: Registration result with metrics
- Registrar: Abstract interface for registration algorithms
- PointToPointICP: Point-to-Point ICP
- PointToPlaneICP: Point-to-Plane ICP
- RegistrationManager: High-level manager with method selection

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import Registrar, RegistrationResult, Transformation
from .manager import RegistrationManager
from .point_to_plane import PointToPlaneICP
from .point_to_point import PointToPointICP

__all__ = [
    "Transformation",
    "RegistrationResult",
    "Registrar",
    "PointToPointICP",
    "PointToPlaneICP",
    "RegistrationManager",
]
