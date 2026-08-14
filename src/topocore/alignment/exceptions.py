"""
topocore.alignment.exceptions
===============================

Exceptions raised by the TopoCore horizontal/vertical alignment
domain.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.core.exceptions import TopoCoreError


class AlignmentError(TopoCoreError):
    """
    Base exception for the alignment domain.
    """


class AlignmentGeometryError(AlignmentError):
    """
    Raised when horizontal/vertical element geometry is degenerate
    or internally inconsistent (e.g. an arc's ``start``/``end`` are
    not at ``radius`` distance from ``center``, or consecutive
    elements in an ``Alignment`` do not chain continuously).
    """


class AlignmentStationError(AlignmentError):
    """
    Raised when a requested station falls outside the range covered
    by an ``Alignment``.
    """


__all__ = [
    "AlignmentError",
    "AlignmentGeometryError",
    "AlignmentStationError",
]
