"""
LAS format support.

This package contains readers and writers for the ASPRS LAS format.

Current API
-----------

- LASReader

Future versions may expose additional helper classes, but only
LASReader is considered part of the public API.
"""

from __future__ import annotations

from .reader import LASReader

__all__ = [
    "LASReader",
]