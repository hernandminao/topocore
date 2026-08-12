"""
topocore.features.catalogs.loaders.csv_loader
==============================================

CSV external feature-code catalog loader.

The CSV format uses a normal comma-separated table. Multiple aliases
inside the ``aliases`` column are separated by semicolons.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import csv
from pathlib import Path

from topocore.features.feature_codes import FeatureCodeDefinition

from .base import CatalogLoadIssue, parse_entries
from .schema import RawCatalogEntry

_REQUIRED_COLUMNS = frozenset(
    {
        "code",
        "name",
        "geometry_type",
        "feature_type",
        "category",
        "layer",
    }
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "si", "sí"})
_FALSE_VALUES = frozenset({"0", "false", "no", "n"})


def _parse_bool_strict(value: str | None, *, column: str, index: int) -> bool:
    """
    Raises
    ------
    ValueError
        If the cell is non-empty and doesn't match a recognized
        true/false representation. An empty cell means "not
        specified" and is legitimately ``False`` -- an unrecognized
        non-empty value (a typo, a stray "verdadero", a "TRUE " with
        a typo) is not silently treated the same way; it's a catalog
        error the author needs to see.
    """
    if value is None or not value.strip():
        return False

    normalized = value.strip().lower()

    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    allowed = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(
        f"Entry {index}: '{column}' has unrecognized value '{value}'. "
        f"Expected one of: {allowed} (or leave the cell empty for false)."
    )


def _parse_aliases(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()

    return tuple(alias.strip() for alias in value.split(";") if alias.strip())


def load_csv(
    path: str | Path,
    *,
    encoding: str = "utf-8-sig",
) -> tuple[FeatureCodeDefinition, ...]:
    """
    Load and validate an external CSV feature-code catalog.

    Expected columns
    ----------------
    code,name,geometry_type,feature_type,category,layer,closed,aliases

    ``closed`` and ``aliases`` are optional columns.
    """
    catalog_path = Path(path)

    with catalog_path.open(
        "r",
        encoding=encoding,
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise ValueError("CSV catalog has no header row.")

        fieldnames = {name.strip() for name in reader.fieldnames if name is not None}

        missing = _REQUIRED_COLUMNS - fieldnames

        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"CSV catalog is missing required column(s): {names}.")

        entries: list[RawCatalogEntry] = []
        issues: list[CatalogLoadIssue] = []

        for index, row in enumerate(reader, start=1):
            feature_type = (row.get("feature_type") or "").strip()

            try:
                closed = _parse_bool_strict(row.get("closed"), column="closed", index=index)
            except ValueError as exc:
                issues.append(
                    CatalogLoadIssue(
                        index=index,
                        code=(row.get("code") or "").strip(),
                        message=str(exc),
                    )
                )
                continue

            entries.append(
                RawCatalogEntry(
                    code=(row.get("code") or "").strip(),
                    name=(row.get("name") or "").strip(),
                    geometry_type=(row.get("geometry_type") or "").strip(),
                    feature_type=feature_type or None,
                    category=(row.get("category") or "").strip(),
                    layer=(row.get("layer") or "").strip(),
                    closed=closed,
                    aliases=_parse_aliases(row.get("aliases")),
                )
            )

    return parse_entries(entries, prior_issues=issues)


__all__ = [
    "load_csv",
]
