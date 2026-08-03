"""
topocore.analysis.profile
=========================

Profile analysis sub-package.

Provides longitudinal, transversal, cross-section, and multi-profile
generation, along with the unified ``ProfileAnalysis`` facade.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .cross_section import (
    CrossSectionProfile,
)
from .longitudinal import (
    LongitudinalProfile,
)
from .manager import (
    ProfileAnalysis,
    ProfileMethod,
)
from .multi_profile import (
    MultiProfile,
)
from .transversal import (
    TransversalProfile,
)

__version__ = "0.1.0"


__all__ = [
    # Version
    "__version__",
    # Manager
    "ProfileAnalysis",
    "ProfileMethod",
    # Generators
    "LongitudinalProfile",
    "TransversalProfile",
    "CrossSectionProfile",
    "MultiProfile",
]
