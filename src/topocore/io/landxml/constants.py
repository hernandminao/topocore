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

#: Decimal places used when formatting coordinates on write. Matches
#: the precision typically emitted by Civil3D/TBC-style exports
#: (see the MicroSurvey/Bentley examples audited for this module).
DEFAULT_COORDINATE_PRECISION: Final[int] = 8

__all__ = [
    "DEFAULT_COORDINATE_PRECISION",
    "LANDXML_1_2_NAMESPACE",
    "LANDXML_VERSION",
    "SURF_TYPE_TIN",
]
