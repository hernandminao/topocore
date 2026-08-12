"""
topocore.features.catalogs.loaders.schema
==========================================

Raw schema used by external feature-code catalog loaders.

The raw representation deliberately keeps semantic enum values as
strings. Conversion into TopoCore domain types is centralized in
``loaders.base`` so JSON, YAML, and CSV catalogs all follow exactly
the same validation path.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RawCatalogEntry:
    """
    Raw external feature-code definition.

    Values are intentionally unparsed. Format-specific loaders are
    responsible only for extracting data; semantic interpretation and
    validation belong to ``parse_entries()``.
    """

    code: str
    name: str
    geometry_type: str
    feature_type: str | None
    category: str
    layer: str
    closed: bool = False
    aliases: tuple[str, ...] = ()


__all__ = [
    "RawCatalogEntry",
]
