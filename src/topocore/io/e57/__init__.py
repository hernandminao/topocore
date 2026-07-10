"""
Support for ASTM E57 point cloud files.

This package provides readers and writers for E57 point cloud files.

Public API
----------

- E57Reader
"""

from __future__ import annotations

from .reader import E57Reader

__all__ = [
    "E57Reader",
]
