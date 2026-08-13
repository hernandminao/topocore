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

from topocore.core.exceptions import TopoCoreError


class ASCIIError(TopoCoreError):
    """
    Base exception for ASCII readers.
    """


class InvalidASCIIRecordError(ASCIIError):
    """
    Raised when an ASCII record cannot be parsed.
    """


class MissingColumnError(ASCIIError):
    """
    Raised when a required column is missing.
    """


class UnsupportedDelimiterError(ASCIIError):
    """
    Raised when a delimiter cannot be detected.
    """
