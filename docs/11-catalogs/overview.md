# `topocore.features.catalogs` — Overview

## Scope, confirmed directly

This is `topocore.features.catalogs` -- a submodule of `topocore.features`
(already closed under `18-features`), not a standalone top-level
`topocore.catalogs` package. It ships the built-in field-survey
feature-code catalog (`ALL_CODES`) and the machinery to load
additional, external catalogs (CSV/JSON/YAML) on top of it. This
audit gives it its own dedicated documentation because it is
substantial in its own right -- 19 files, 2925 lines, larger than
`19-gpkg` (1161) or `20-dxf` (1009) -- and because it directly
participates in `Feature Extraction`'s own public contract
(`FeatureCodeRegistry`, consumed by `Workflow.build_features_from_survey()`).

## Real inventory — 19 files, 2925 lines

```text
topocore/features/catalogs/
├── __init__.py         wiring point: assembles 9 domain catalogs into ALL_CODES
├── _validation.py         validate_definition_geometry() -- geometry invariant
├── _pending.py              documented, deferred survey codes (not yet migrated)
├── catalog_audit.py          run_audit() -- self-audit tooling (see below)
├── cadastre.py, control.py, default.py, drainage.py, structures.py,
│   terrain.py, transportation.py, utilities.py, vegetation.py
│                             the 9 domain catalogs, 160 real codes total
└── loaders/
    ├── __init__.py
    ├── schema.py            RawCatalogEntry -- unparsed, format-agnostic
    ├── base.py                build_raw_entry_from_mapping() / parse_entries() --
    │                          shared strict-typing + semantic validation,
    │                          used identically by json_loader and yaml_loader
    ├── csv_loader.py           load_csv()
    ├── json_loader.py           load_json()
    └── yaml_loader.py            load_yaml() (PyYAML optional, imported lazily)
```

`FeatureCodeDefinition`/`FeatureCodeRegistry` themselves live in
`topocore.features.feature_codes`, not in this submodule -- `catalogs`
supplies data (`ALL_CODES`) and external-loading machinery; the
registry type itself is shared, general-purpose infrastructure
`feature_builder.py` also depends on directly.

## What a catalog code represents

A `FeatureCodeDefinition` separates 4 orthogonal concepts, confirmed
by direct reading and by real execution: `code` (the survey field
code itself), `feature_type`/`category` (the semantic `FeatureType`
this code produces), `geometry_type` (how consecutive same-code
points become geometry -- `POINT`/`SYMBOL`/`LINE`/`POLYGON`/`GROUND`),
and `layer` (default CAD/GIS presentation). `GROUND`-geometry codes
are the one deliberate exception: they declare no `feature_type` at
all (enforced by `_validation.py`, not by the dataclass itself) --
bare terrain shots feed TIN/DTM construction directly and never
produce a `Feature`.

## The self-audit already built into this module

`catalog_audit.run_audit(ALL_CODES)`, executed directly during this
audit: **160 codes, 0 violations, PASS** -- confirming this module
already validates its own internal consistency (duplicate detection,
geometry-invariant compliance) as part of its own real, existing
tooling, not something this audit had to build.

## Integration — confirmed consistent in both directions

- **`catalogs` -> `features`**: an unregistered survey code is never
  an exception -- `feature_builder.py`'s own `build()` records a
  `BuildDiagnostic` (reason `UNREGISTERED_CODE`) and routes the
  points to `unmatched`, confirmed directly. `FeatureType` has no
  obligatory dependency on any `FeatureCodeDefinition` existing at
  all -- a code without a matching definition simply produces no
  `Feature`, not a failure.
- **`catalogs` -> `workflow`**: `Workflow.build_features_from_survey()`
  uses exactly the registry it is given (`FeatureCodeRegistry.default()`
  or a caller-supplied one), confirmed directly to add no additional
  guarantee of its own -- `Workflow` introduces no wrapping, no
  extra validation, and no different behavior than calling
  `FeatureBuilder` directly would.
- **The 8 `FeatureType` members with zero catalog code**
  (`breakline`, `contour`, `slope_change`, `drainage`, `road`,
  `parking`, `driveway`, `sign`): reconfirmed identically from this
  module's own side (`ALL_CODES` itself), matching `18-features`'s
  own finding exactly, with no discrepancy. Confirmed deliberate, per
  the project's own author: catalog codes exist only for
  field-surveyed features (total station/GNSS, where a person assigns
  a code per point); these 8 are detected only from point-cloud
  geometry/classification or TIN analysis, never from a field code.

## The one real finding of this audit's own — not a defect

`FeatureCodeRegistry.register_many()` is not atomic: if one entry in
a batch collides with an already-registered, differently-defined
code, entries registered before the collision remain registered.
Investigated thoroughly (see [`validation.md`](./validation.md) for
the complete real-execution evidence) and confirmed:

- No documentation or docstring anywhere promises all-or-nothing
  semantics for this method.
- The only 2 real, current call sites in this codebase
  (`FeatureCodeRegistry.__init__()`'s own convenience wrapper, and
  `default()`'s own call with the already-validated `ALL_CODES`)
  never exercise the partial-failure path at all.
- The partiality is confirmed benign, not corruptive: a
  pre-existing code's own definition is never overwritten by a
  colliding attempt (confirmed with a real, executed reproduction);
  the registry remains fully usable afterward; a caller can retry
  cleanly with the non-conflicting subset.

Classified as a documented, non-corruptive behavioral note, not a
defect -- see [`limitations.md`](./limitations.md) for the complete
framing.

## Result of this audit: 0 defects, 0 code changes

Matching `21-workflow`'s own outcome, not `18-features`/`19-gpkg`/
`20-dxf`'s: this audit found no functional defect requiring
correction in `topocore.features.catalogs`, and made no code change.
The same 5-point methodology was applied in full, including targeted
real-execution reproductions of every error path a catalog author or
a `Workflow` caller could realistically encounter -- see
[`validation.md`](./validation.md) for the complete account.

## Where to go next

- [`validation.md`](./validation.md) -- the complete real-execution
  evidence for Points 1-5, including the `register_many()` scenario
  in full.
- [`limitations.md`](./limitations.md) -- the `register_many()`
  behavioral note, and the 1 pre-existing `mypy` finding.
