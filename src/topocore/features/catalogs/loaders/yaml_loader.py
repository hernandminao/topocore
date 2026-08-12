"""
topocore.features.catalogs.loaders.yaml_loader
===============================================

YAML external feature-code catalog loader.

PyYAML is an optional dependency and is imported lazily so TopoCore
does not require YAML support unless this loader is actually used.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from topocore.features.feature_codes import FeatureCodeDefinition

from .base import CatalogLoadIssue, build_raw_entry_from_mapping, parse_entries
from .schema import RawCatalogEntry


def _require_yaml() -> ModuleType:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML catalog support requires PyYAML. Install TopoCore with the 'yaml' optional dependency."
        ) from exc

    return yaml


def load_yaml(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> tuple[FeatureCodeDefinition, ...]:
    """Load and validate an external YAML feature-code catalog."""
    yaml = _require_yaml()
    catalog_path = Path(path)

    with catalog_path.open("r", encoding=encoding) as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, dict):
        raise TypeError("Catalog root must be a YAML mapping.")

    codes = payload.get("codes")

    if not isinstance(codes, list):
        raise TypeError("Catalog must contain a 'codes' sequence.")

    entries: list[RawCatalogEntry] = []
    issues: list[CatalogLoadIssue] = []

    for index, item in enumerate(codes, start=1):
        if not isinstance(item, dict):
            issues.append(
                CatalogLoadIssue(
                    index=index,
                    code="<unknown>",
                    message=f"Catalog entry {index} must be a YAML mapping, got {type(item).__name__}.",
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
    "load_yaml",
]
