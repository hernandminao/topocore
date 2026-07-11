"""
topocore.features.catalogs
=============================

Feature-code catalogs shipped with TopoCore.

Each catalog is a plain tuple of ``FeatureCodeDefinition`` in its own
module (``default``, ``terrain``, and future ones such as roads,
utilities, or vegetation). ``ALL_CODES`` is the concatenation of all
of them, and is what ``FeatureCodeRegistry.default()`` loads.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.feature_codes import FeatureCodeDefinition

from .default import DEFAULT_CODES
from .terrain import TERRAIN_CODES

ALL_CODES: tuple[FeatureCodeDefinition, ...] = (
    *DEFAULT_CODES,
    *TERRAIN_CODES,
)

__all__ = [
    "DEFAULT_CODES",
    "TERRAIN_CODES",
    "ALL_CODES",
]
