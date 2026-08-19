"""
topocore.analysis.quality
=========================

Quality analysis sub-package.

Provides RMSE, GPS control, precision, registration, completeness,
correctness, Cloud-to-Cloud, Cloud-to-Mesh, Hausdorff, and Chamfer
distance quality assessments.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .c2c import CloudToCloudDistance
from .c2m import CloudToMeshDistance
from .chamfer import ChamferDistance
from .completeness import CompletenessAnalysis
from .correctness import CorrectnessAnalysis
from .gps_control import GPSControlAnalysis
from .hausdorff import HausdorffDistance
from .manager import QualityAnalysis, QualityMethod
from .precision import PrecisionAnalysis
from .registration import RegistrationQuality
from .rmse import RMSEAnalysis

__all__ = [
    "ChamferDistance",
    "CloudToCloudDistance",
    "CloudToMeshDistance",
    "CompletenessAnalysis",
    "CorrectnessAnalysis",
    "GPSControlAnalysis",
    "HausdorffDistance",
    "PrecisionAnalysis",
    "QualityAnalysis",
    "QualityMethod",
    "RMSEAnalysis",
    "RegistrationQuality",
]
