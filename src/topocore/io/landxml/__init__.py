"""
topocore.io.landxml
=====================

LandXML 1.2 IO for TopoCore: ``<Surfaces>``/TIN, ``<CgPoints>``, and
``<Alignments>`` (``<CoordGeom>`` + ``<Profile>``/``<ProfAlign>``).
``<Feature>`` (embedded in ``<CgPoint>``) remains out of scope -- see
``topocore.io.landxml.models`` for the reasoning.

Alignment scope: ``<Curve crvType="arc">`` and ``<Spiral spiType="clothoid">``
only; chord-defined curves, other spiral types, curve-to-curve
compound spirals, and ``<CircCurve>`` vertical curves are skipped
with an explicit warning in the read report, not silently dropped
or hard failures.

Coordinate convention
----------------------
LandXML coordinate text is always "north east elev"; see
``topocore.io.landxml.coordinates`` for the mapping to
``Point3D``/``SurveyPoint``, and ``topocore.io.landxml.codecs`` for
the "INF" radius, "cw"/"ccw" rotation, and "station elevation"
conventions used by ``<Alignments>``.

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
    NamedAlignment,
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
    "NamedAlignment",
    "NamedPointGroup",
    "NamedSurface",
]
