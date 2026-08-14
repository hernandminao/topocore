"""
topocore.io.landxml.constants
==============================

Shared constants for the TopoCore LandXML IO subsystem.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final

#: XML namespace for LandXML 1.2, the version this subsystem reads
#: and writes. Older (1.0/0.88) and newer (2.0) namespaces are not
#: recognized on read -- see ``reader.py``.
# XML namespace identifier (not a network URL); S5332 does not apply.
LANDXML_1_2_NAMESPACE: Final[str] = "http://www.landxml.org/schema/LandXML-1.2"  # NOSONAR

#: ``version`` attribute written on the root ``<LandXML>`` element.
LANDXML_VERSION: Final[str] = "1.2"

#: Surface definition type this subsystem supports. LandXML also
#: defines "GRID" and "ROAD" surface types; both are out of scope
#: (see PR18B contract -- TIN only).
SURF_TYPE_TIN: Final[str] = "TIN"

#: Decimal places used when formatting coordinates on write. 10 (not
#: 8, the precision commonly seen in real Civil3D/TBC exports) --
#: SpiralElement's chord-consistency check uses an absolute tolerance
#: of 1e-9 (topocore.math.config.DEFAULT_MATH_CONFIG), and 8 decimal
#: places alone can introduce ~3e-9 of rounding error on a
#: hundreds-magnitude coordinate, enough to make a perfectly valid
#: spiral round-trip fail reconstruction. 10 decimal places keeps
#: rounding error under ~5e-11, safely below that tolerance with
#: margin. Confirmed empirically, not assumed -- see the PR18C
#: session notes.
DEFAULT_COORDINATE_PRECISION: Final[int] = 10

#: Curve geometry type this subsystem supports for <Curve>. LandXML
#: also allows chord-defined / three-point curves without an
#: explicit radius+center; out of scope -- see PR18C contract.
CRV_TYPE_ARC: Final[str] = "arc"

#: Transition curve type this subsystem supports for <Spiral>.
#: LandXML also allows "cubic", "bloss", "biquadraticParabola", etc.;
#: SpiralElement models a true Euler spiral (clothoid) specifically,
#: so only this type is accepted -- see PR18C contract.
SPI_TYPE_CLOTHOID: Final[str] = "clothoid"

#: Geometric consistency tolerance used ONLY when constructing
#: ArcElement/SpiralElement from parsed LandXML data (via their
#: optional ``tolerance`` constructor parameter) -- an
#: interoperability tolerance for serialized engineering data, NOT
#: TopoCore's domain precision (topocore.math.config.DEFAULT_MATH_CONFIG,
#: which stays untouched).
#:
#: Confirmed empirically against TWO independent real-world sources
#: (PR18C session):
#:   - Autodesk Civil 3D 2007 (GSG_features_alignments.xml, 47 real
#:     <Curve> elements, 8-10 decimal place coordinates): max
#:     observed Start/Center/End-vs-radius deviation 7.995e-9.
#:   - PLATEIA 2007 (Sample_Plateia2007LandXML11.XML, 6 real
#:     <Spiral> elements, 6 decimal place coordinates): max observed
#:     chord-length deviation 1.083e-6.
#:
#: Root cause confirmed to be the SAME phenomenon in both cases --
#: decimal-place text truncation of Start/End/Center/PI, scaling
#: with each producer's own export precision (6 vs 8-10 decimals) --
#: not a property of Arc vs. Spiral, and not something that needs a
#: separate tolerance per element type. 2e-6 (not 1e-6) gives a
#: small margin above the largest real deviation observed so far
#: (1.083e-6). Deliberately a single fixed constant, NOT derived
#: from a given file's own decimal precision: decimal-place count
#: alone does not guarantee a matching geometric uncertainty, so
#: auto-deriving it per file would be a false precision. Re-audit
#: against further real files before assuming this holds universally.
LANDXML_GEOMETRY_TOLERANCE: Final[float] = 2e-6

__all__ = [
    "CRV_TYPE_ARC",
    "DEFAULT_COORDINATE_PRECISION",
    "LANDXML_1_2_NAMESPACE",
    "LANDXML_GEOMETRY_TOLERANCE",
    "LANDXML_VERSION",
    "SPI_TYPE_CLOTHOID",
    "SURF_TYPE_TIN",
]
