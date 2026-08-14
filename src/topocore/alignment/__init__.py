"""
topocore.alignment
====================

Horizontal and vertical alignment domain for TopoCore. Built to
support LandXML ``<Alignments>``/``<Profile>`` interoperability --
see ``topocore.io.landxml`` -- but is a first-class geometric domain
in its own right, independent of the LandXML format.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.alignment.elements import (
    ArcElement,
    HorizontalElement,
    LineElement,
    SpiralElement,
)
from topocore.alignment.exceptions import (
    AlignmentError,
    AlignmentGeometryError,
    AlignmentStationError,
)
from topocore.alignment.models import Alignment, DesignProfile
from topocore.alignment.vertical_elements import (
    GradeSegment,
    VerticalCurve,
    VerticalElement,
)

__all__ = [
    "Alignment",
    "AlignmentError",
    "AlignmentGeometryError",
    "AlignmentStationError",
    "ArcElement",
    "DesignProfile",
    "GradeSegment",
    "HorizontalElement",
    "LineElement",
    "SpiralElement",
    "VerticalCurve",
    "VerticalElement",
]
