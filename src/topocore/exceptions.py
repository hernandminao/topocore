"""
topocore.exceptions
===================

Base exceptions for the TopoCore package.

All exceptions raised by TopoCore inherit from TopoCoreError.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations


class TopoCoreError(Exception):
    """
    Base exception for every error raised by TopoCore.
    """


__all__ = [
    "TopoCoreError",
]
