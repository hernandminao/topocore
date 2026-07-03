"""
TopoCore Math Module.

This package provides the numerical foundation for the entire
TopoCore geometry engine.
"""

from .config import DEFAULT_MATH_CONFIG, MathConfig
from .constants import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_DECIMAL_PRECISION,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RELATIVE_TOLERANCE,
    EPSILON,
    MAX_COORDINATE_MAGNITUDE,
)
from .numeric import (
    clamp,
    cube,
    lerp,
    mean,
    safe_divide,
    sign,
    square,
)
from .tolerance import (
    is_positive,
    is_close,
    is_zero,
    is_negative,
)
from .validation import (
    validate_coordinate,
    validate_finite,
    validate_not_nan,
    validate_positive,
)

__all__ = [
    "MathConfig",
    "DEFAULT_MATH_CONFIG",
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "EPSILON",
    "DEFAULT_DECIMAL_PRECISION",
    "DEFAULT_MAX_ITERATIONS",
    "MAX_COORDINATE_MAGNITUDE",
    "clamp",
    "cube",
    "lerp",
    "mean",
    "safe_divide",
    "sign",
    "square",
    "is_positive",
    "is_close",
    "is_zero",
    "is_negative",
    "validate_coordinate",
    "validate_finite",
    "validate_not_nan",
    "validate_positive",
]