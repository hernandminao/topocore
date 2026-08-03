"""
topocore.processing.backend
===========================

Mathematical backend abstraction layer.

This package provides a pluggable backend system that allows TopoCore
to use different array libraries (NumPy, CuPy, Torch, JAX, etc.)
without modifying the algorithm implementations.

The backend system follows the Strategy pattern: algorithms receive a
Backend instance and delegate all mathematical operations to it.

Public API
----------
- Backend: abstract base class defining the interface.
- NumPyBackend: default implementation using NumPy.
- NUMPY_BACKEND: singleton instance of the NumPy backend.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import Backend
from .numpy_backend import NUMPY_BACKEND, NumPyBackend

__all__ = [
    "Backend",
    "NumPyBackend",
    "NUMPY_BACKEND",
]
