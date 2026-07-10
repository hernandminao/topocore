"""
topocore.io.ply.exceptions
==========================

Exceptions raised by the PLY reader.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.io.exceptions import InvalidHeaderError


class PLYError(InvalidHeaderError):
    """
    Base class for all PLY-related errors.
    """


class InvalidPLYError(PLYError):
    """
    Raised when a PLY file violates the PLY specification.
    """


__all__ = [
    "PLYError",
    "InvalidPLYError",
]
