"""
topocore.io.landxml.exceptions
===============================

Exceptions raised by the TopoCore LandXML IO subsystem.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.core.exceptions import TopoCoreError


class LandXMLError(TopoCoreError):
    """
    Base exception for all LandXML IO errors.
    """


class LandXMLParseError(LandXMLError):
    """
    Raised when a LandXML file is malformed or a required element or
    attribute is missing.
    """


class LandXMLValidationError(LandXMLError):
    """
    Raised when a parsed or about-to-be-written LandXML document
    fails semantic validation (e.g. a ``<F>`` face referencing a
    point id absent from ``<Pnts>``, duplicated point ids,
    non-finite coordinates).
    """


class LandXMLWriteError(LandXMLError):
    """
    Raised when serializing a ``LandXMLDocument`` to a file fails.
    """


__all__ = [
    "LandXMLError",
    "LandXMLParseError",
    "LandXMLValidationError",
    "LandXMLWriteError",
]
