"""
topocore.features.catalogs.loaders.json_loader
===============================================

JSON external feature-code catalog loader.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import json
from pathlib import Path

from topocore.features.feature_codes import FeatureCodeDefinition

from .base import CatalogLoadIssue, build_raw_entry_from_mapping, parse_entries
from .schema import RawCatalogEntry


def load_json(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> tuple[FeatureCodeDefinition, ...]:
    """Load and validate an external JSON feature-code catalog."""
    catalog_path = Path(path)

    with catalog_path.open("r", encoding=encoding) as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise TypeError("Catalog root must be a JSON object.")

    codes = payload.get("codes")

    if not isinstance(codes, list):
        raise TypeError("Catalog must contain a 'codes' array.")

    entries: list[RawCatalogEntry] = []
    issues: list[CatalogLoadIssue] = []

    for index, item in enumerate(codes, start=1):
        if not isinstance(item, dict):
            issues.append(
                CatalogLoadIssue(
                    index=index,
                    code="<unknown>",
                    message=f"Catalog entry {index} must be a JSON object, got {type(item).__name__}.",
                )
            )
            continue

        try:
            entries.append(build_raw_entry_from_mapping(item))
        except (ValueError, TypeError) as exc:
            issues.append(
                CatalogLoadIssue(
                    index=index,
                    code=str(item.get("code", "")),
                    message=str(exc),
                )
            )

    return parse_entries(entries, prior_issues=issues)


__all__ = [
    "load_json",
]
