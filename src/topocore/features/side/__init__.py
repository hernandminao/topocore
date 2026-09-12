"""
topocore.features.side
=========================

Left/right laterality resolution for linear features (pavement
edges, by default) relative to a reference CENTERLINE. See
`topocore.features.side.resolver` for the full design rationale.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.side.exceptions import SideResolutionError
from topocore.features.side.models import Side, SideMethod
from topocore.features.side.resolver import SideResolver

__all__ = [
    "Side",
    "SideMethod",
    "SideResolutionError",
    "SideResolver",
]
