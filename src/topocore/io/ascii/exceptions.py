"""
topocore.io.ascii.exceptions
============================

Exceptions raised by ASCII readers.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations


class ASCIIError(Exception):
    """
    Base exception for ASCII readers.
    """


class InvalidASCIIRecordError(ASCIIError):
    """
    Raised when an ASCII record cannot be parsed.
    """


class InvalidHeaderError(ASCIIError):
    """
    Raised when a header cannot be interpreted.
    """


class MissingColumnError(ASCIIError):
    """
    Raised when a required column is missing.
    """


class UnsupportedDelimiterError(ASCIIError):
    """
    Raised when a delimiter cannot be detected.
    """