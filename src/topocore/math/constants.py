"""
topocore.math.constants
=======================

Global mathematical constants used by TopoCore.

Only constants specific to the engine should be defined here.
Do not redefine constants available in Python's standard library
(e.g. math.pi or math.e).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final

###############################################################################
# Floating-point comparison
###############################################################################

#: Default absolute tolerance for floating-point comparisons.
DEFAULT_ABSOLUTE_TOLERANCE: Final[float] = 1e-9

#: Default relative tolerance for floating-point comparisons.
DEFAULT_RELATIVE_TOLERANCE: Final[float] = 1e-12

###############################################################################
# Numerical precision
###############################################################################

#: Values whose absolute value is below EPSILON are considered numerically zero.
EPSILON: Final[float] = 1e-12

###############################################################################
# Formatting
###############################################################################

#: Default number of decimal places when exporting values.
DEFAULT_DECIMAL_PRECISION: Final[int] = 9

###############################################################################
# Coordinate validation
###############################################################################

#: Maximum accepted coordinate magnitude.
MAX_COORDINATE_MAGNITUDE: Final[float] = 1.0e12

###############################################################################
# Algorithms
###############################################################################

#: Default maximum number of iterations for iterative algorithms.
DEFAULT_MAX_ITERATIONS: Final[int] = 1000

###############################################################################
# Public API
###############################################################################

__all__ = [
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "EPSILON",
    "DEFAULT_DECIMAL_PRECISION",
    "MAX_COORDINATE_MAGNITUDE",
    "DEFAULT_MAX_ITERATIONS",
]