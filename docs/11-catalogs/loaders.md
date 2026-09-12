# Catalog Loaders — Shared Contract

`topocore.features.catalogs.loaders` provides three format-specific
entry points — `load_json`, `load_csv`, `load_yaml` — that all
converge on exactly the same validation path, so a catalog behaves
identically regardless of which format it's written in.

```text
Format-specific file
         │
         ▼
┌─────────────────────────────┐
│ Stage 1: raw extraction     │   json_loader / csv_loader / yaml_loader
│ (strict types, no coercion) │
└──────────────┬──────────────┘
               │
               ▼
        RawCatalogEntry
               │
               ▼
┌─────────────────────────────┐
│ Stage 2: semantic parsing   │   loaders.base.parse_entries()
│ (enum resolution, geometry  │
│  invariant)                 │
└──────────────┬──────────────┘
               │
               ▼
      FeatureCodeDefinition
```

## Stage 1 — raw extraction, strict types

Each loader reads its own file format into a `RawCatalogEntry` — a
dataclass that intentionally keeps every semantic field as a raw
string. This stage does *not* try to be helpful by guessing at
intent: a `closed` field must be an actual boolean, not a string that
merely looks like one.

This matters concretely: in Python, `bool("false")` is `True`,
because any non-empty string is truthy. A loader that coerced
strings would turn a catalog author's `"false"` into an active
`True` — a silently wrong, hard-to-spot catalog error. TopoCore's
loaders reject this outright rather than coerce it.

`json_loader` and `yaml_loader` share the exact same extraction code
(`build_raw_entry_from_mapping`), since both parse into the same
Python `dict` shape. `csv_loader` has its own, slightly different
extraction step because CSV cells are always strings and a
recognized set of true/false spellings is accepted (see
[`csv-loader.md`](./csv-loader.md)).

## Stage 2 — semantic parsing

`parse_entries()` takes the raw entries and:

1. Resolves `geometry_type`, `feature_type`, and `category` strings
   into their corresponding enum members (`FeatureGeometryType`,
   `FeatureType`, `FeatureCategory`), case-insensitively.
2. Rejects any value that isn't one of TopoCore's own existing enum
   members — a custom catalog remaps field codes onto TopoCore's
   ontology, it does not extend it.
3. Enforces the geometry invariant (a `feature_type` is required
   for every non-`GROUND` geometry).
4. Constructs the final, validated `FeatureCodeDefinition`.

## Error reporting: every problem, not just the first

Both stages collect **every** problem they find rather than stopping
at the first one, and a single `ExternalCatalogError` reports all of
them together — from both stages combined:

```python
>>> from topocore.features.catalogs.loaders.json_loader import load_json
>>> load_json("catalog_with_three_errors.json")
Traceback (most recent call last):
    ...
ExternalCatalogError: External feature-code catalog contains 3 error(s):
  [1] BADCODE1: Unknown geometry_type 'squiggle'. Expected one of: ground, line, point, polygon, symbol.
  [2] BADCODE2: Unknown feature_type 'not_a_real_type'. External catalogs remap field codes onto TopoCore's existing ontology; they do not introduce new FeatureType values.
  [3] BADCODE3: Code 'BADCODE3' must declare a non-empty layer.
```

This lets a catalog author fix an entire file in one pass instead of
re-running the loader after each single fix.

## `ExternalCatalogError`

```python
class ExternalCatalogError(ValueError):
    issues: tuple[CatalogLoadIssue, ...]
```

Each `CatalogLoadIssue` carries the 1-based `index` of the failing
entry, its `code` (or `"<empty>"`/`"<unknown>"` when even that
couldn't be extracted), and a human-readable `message`.

## Format-specific reference

- [`json-loader.md`](./json-loader.md)
- [`csv-loader.md`](./csv-loader.md)
- [`yaml-loader.md`](./yaml-loader.md)
