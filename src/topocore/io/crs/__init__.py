"""
topocore.io.crs
==================

Shared, format-agnostic CRS detection: `ExternalCRSDetector`, for
formats that carry no CRS of their own (XYZ, CSV, PTS, PLY) and rely
on an external `.prj` sidecar file instead. Kept separate from the
format-specific detectors (`topocore.io.las.crs_detection`,
`topocore.io.e57.crs_detection`, `topocore.io.landxml.crs_detection`)
since this one operates the same way regardless of which of the 4
formats the main data file actually is.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.io.crs.detector import detect_crs
from topocore.io.crs.external import (
    ExternalCRSDetector,
    apply_crs_with_native_priority,
    apply_external_crs,
)

__all__ = [
    "ExternalCRSDetector",
    "apply_crs_with_native_priority",
    "apply_external_crs",
    "detect_crs",
]
