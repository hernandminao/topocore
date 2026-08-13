"""
topocore.io.landxml
=====================

LandXML 1.2 IO for TopoCore: ``<Surfaces>``/TIN and ``<CgPoints>``
only (PR18B). ``<Alignments>``, ``<Profile>`` and ``<Feature>``
(embedded in ``<CgPoint>``) are out of scope -- see
``topocore.io.landxml.models`` for the reasoning.

Coordinate convention
----------------------
LandXML coordinate text is always "north east elev"; see
``topocore.io.landxml.coordinates`` for the mapping to
``Point3D``/``SurveyPoint``.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.io.landxml.exceptions import (
    LandXMLError,
    LandXMLParseError,
    LandXMLValidationError,
    LandXMLWriteError,
)
from topocore.io.landxml.models import (
    LandXMLDocument,
    LinearUnit,
    NamedPointGroup,
    NamedSurface,
)
from topocore.io.landxml.reader import LandXMLReader
from topocore.io.landxml.report import LandXMLReadReport, LandXMLWriteReport
from topocore.io.landxml.validation import LandXMLValidator
from topocore.io.landxml.writer import LandXMLWriter

__all__ = [
    "LandXMLDocument",
    "LandXMLError",
    "LandXMLParseError",
    "LandXMLReadReport",
    "LandXMLReader",
    "LandXMLValidationError",
    "LandXMLValidator",
    "LandXMLWriteError",
    "LandXMLWriteReport",
    "LandXMLWriter",
    "LinearUnit",
    "NamedPointGroup",
    "NamedSurface",
]
