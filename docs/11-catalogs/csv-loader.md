# CSV Catalog Loader

```python
from topocore.features.catalogs.loaders.csv_loader import load_csv

definitions = load_csv("my_catalog.csv", encoding="utf-8-sig")
```

No optional dependency — uses Python's standard library `csv`
module. Default encoding is `utf-8-sig` (tolerates a leading BOM,
which spreadsheet software commonly adds when saving CSV).

## File shape

A plain comma-separated table with a header row:

```csv
code,name,geometry_type,feature_type,category,layer,closed,aliases
MICERCA,Custom fence,line,fence,infrastructure,MY_LAYER,false,CERCA_ALT
```

Required columns: `code`, `name`, `geometry_type`, `feature_type`,
`category`, `layer`. `closed` and `aliases` are optional columns —
if omitted entirely, every row defaults to `closed=false` and no
aliases.

**Multiple aliases** in the `aliases` column are separated by
semicolons: `ALIAS_ONE;ALIAS_TWO`.

## Accepted boolean spellings for `closed`

Unlike JSON/YAML (which use native booleans), a CSV cell is always a
string. `closed` accepts, case-insensitively:

- **True**: `1`, `true`, `yes`, `y`, `si`, `sí`
- **False**: `0`, `false`, `no`, `n`
- **Empty cell**: also `false` (means "not specified")

Any other non-empty value is rejected — a typo like `verdadero` is
not silently treated as `false`:

```python
>>> load_csv("catalog_with_typo.csv")
Traceback (most recent call last):
    ...
ExternalCatalogError: External feature-code catalog contains 1 error(s):
  [1] X: Entry 1: 'closed' has unrecognized value 'verdadero'. Expected
      one of: 0, 1, false, n, no, si, sí, true, y, yes (or leave the
      cell empty for false).
```

## Structural errors (raised immediately)

```python
>>> load_csv("empty_file.csv")
ValueError: CSV catalog has no header row.

>>> load_csv("missing_column.csv")   # e.g. no 'feature_type' column
ValueError: CSV catalog is missing required column(s): feature_type.
```

## Per-entry errors

As with JSON and YAML, semantic problems (unknown enum values, empty
required fields) across every row are collected and reported
together as a single `ExternalCatalogError` — see
[`loaders.md`](./loaders.md) for the shared mechanism.
