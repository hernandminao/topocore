# Validation — 0 defects found, complete real-execution evidence

## Methodology, identical to `18-features`/`19-gpkg`/`20-dxf`/`21-workflow`

```text
Point 1: real inventory       -> Point 2: contracts/consumers ->
Point 3: integration          -> Point 4: error paths         ->
Point 5: regression
```

As with `21-workflow`, this methodology did not lead to a defect
here. This document records the real-execution evidence gathered,
including the one behavioral observation investigated in full before
being classified as not a defect.

## Registry API — confirmed by direct execution

- `register()` on a duplicate code with a **different** definition:
  confirmed rejected with `ValueError`, naming both the existing and
  the new definition.
- `register()` on a duplicate code with an **identical** definition:
  confirmed accepted (idempotent) -- no exception, registry size
  unchanged. This distinction (identity comparison, not merely
  key collision) is confirmed load-bearing for the `register_many()`
  scenario below.
- `get()` on an unregistered code: confirmed returns `None`, not an
  exception -- there is no `contains()` method; `get() is not None`
  is this codebase's own real idiom, confirmed by its one real
  consumer (`feature_builder.py`).

## Geometry validation boundary — confirmed by direct execution, both paths

- `validate_definition_geometry()` called directly (its own
  documented "explicit validation/test utility" role): confirmed to
  raise `CatalogGeometryError` unwrapped, for a genuinely
  incompatible combination (`FeatureType.SIGN` -- which
  `_EXPECTED_GEOMETRY` only allows as `POINT` -- declared with
  `geometry_type=POLYGON`).
- The identical incompatible entry, submitted through `load_json()`
  instead: confirmed the same underlying check is folded into
  `ExternalCatalogError`'s own issue list (since `CatalogGeometryError`
  is itself a `ValueError`, caught by `parse_entries()`'s own
  `except ValueError`) -- a single, unified exception type for every
  loader-detected problem, geometry-related or not.
- A structurally compatible combination was confirmed NOT to trigger
  this check: `FeatureType.TREE` genuinely allows both `POINT` and
  `POLYGON` (a tree crown outline is a legitimate representation) --
  confirmed directly from `_EXPECTED_GEOMETRY` before selecting a
  genuinely incompatible test case, avoiding a false-positive
  reproduction.

## Loaders — every real error path confirmed

- Non-existent file (`load_json`): confirmed raises `FileNotFoundError`
  unwrapped.
- Empty file (`load_json`): confirmed raises `json.JSONDecodeError`
  unwrapped.
- Malformed JSON syntax: confirmed raises `json.JSONDecodeError`
  unwrapped.
- Unknown `feature_type` value: confirmed collected into
  `ExternalCatalogError`'s own issue list, naming the exact bad value.
- CSV missing required columns: confirmed raises `ValueError` naming
  every missing column by name (`category, feature_type, geometry_type,
  layer`), not just the first one found.
- Unicode content (a code and name containing `Ñ`/accented
  characters): confirmed round-trips correctly through `load_json()`.

The unwrapped `FileNotFoundError`/`JSONDecodeError` cases are
confirmed NOT analogous to the exception-boundary defects found in
`18-features`/`19-gpkg`/`20-dxf`: those 3 fixes were about exceptions
escaping a `Workflow`/exporter's own internal `strict`-aware
recovery boundary. `load_json()`/`load_csv()`/`load_yaml()` are
confirmed, by direct search (see [`overview.md`](./overview.md)), to
never be called through any such boundary -- they are standalone
utilities an application calls directly, the same way calling
`json.load()` itself would raise its own natural exceptions.

## `register_many()`'s own non-atomicity — the complete real scenario

The specific scenario this audit's own Point 4 focused on, executed
directly against the real, built-in default registry:

```text
FeatureCodeRegistry.default()
      -> 159 codes (the real, current size of ALL_CODES)

An external, valid JSON catalog (3 entries, loaded atomically via
load_json() -- confirmed the loader itself returned exactly 3
FeatureCodeDefinition, no partial result possible at this stage):
      -> 2 new, non-conflicting codes ("NUEVO1", "NUEVO2")
      -> 1 entry using an already-registered code ("CERCA"), with a
         conflicting definition

registry.register_many(catalogo_externo)
      -> ValueError raised, naming the conflicting code

Confirmed immediately afterward:
      -> registry now holds 161 codes (159 + 2) -- NUEVO1 and NUEVO2
         are both registered
      -> the pre-existing "CERCA" code's own original definition
         (name "Fence") is confirmed UNCHANGED -- the conflicting
         attempt never overwrote it
      -> the registry remains fully usable: get() on any known code,
         including "CERCA", still returns the correct definition
      -> a retry with only the non-conflicting subset (the 2 valid
         entries, with the conflicting one filtered out) succeeds
         cleanly
```

**Classification, per the evidence above**: this is confirmed a
real, genuine behavior -- `register_many()` is not atomic -- but
confirmed NOT to violate any documented contract (no docstring
anywhere promises all-or-nothing semantics) and confirmed NOT to
corrupt anything (pre-existing definitions are protected by the same
identity check that makes idempotent re-registration safe; only
new, non-conflicting codes are added before the failure). The 2 real
call sites in this codebase (`__init__()`'s own convenience wrapper,
`default()`'s own call with the already-validated `ALL_CODES`) never
exercise this path at all in practice. Recorded in
[`limitations.md`](./limitations.md) as a documented behavioral note,
not corrected as a defect.

## Integration — confirmed consistent, both directions

- An unregistered survey code reaching `feature_builder.build()`:
  confirmed produces a `BuildDiagnostic` (`UNREGISTERED_CODE`), not
  an exception -- the affected points are routed to `unmatched`.
- `Workflow.build_features_from_survey()`: confirmed, by reading its
  own source, to add no validation or wrapping of its own around
  whichever registry it is given (`default()` or caller-supplied) --
  it behaves identically to calling `FeatureBuilder` directly.
- The 8 `FeatureType` members with no catalog code
  (`breakline`/`contour`/`slope_change`/`drainage`/`road`/`parking`/
  `driveway`/`sign`): reconfirmed identically from `ALL_CODES` itself,
  with the exact same 8 names already found during `18-features`'s
  own audit -- no discrepancy between the 2 vantage points.

## Regression

- `ruff`, with the real project's own `pyproject.toml`: 0 findings
  anywhere in `topocore/features/catalogs/`.
- `mypy`, same real configuration, isolated to this submodule's own
  19 files (`--follow-imports=silent`): 1 finding --
  `loaders/yaml_loader.py`'s own `_require_yaml()`, `no-any-return`
  on `return yaml` (the imported module itself is `Any`-typed,
  since `mypy` cannot fully resolve `PyYAML`'s own stubs through a
  runtime `import` inside a function body). The same category of
  finding already documented multiple times elsewhere in this project
  (`16-terrain`, `19-gpkg` -- third-party stub imprecision, not a
  runtime defect). Not corrected, consistent with this audit's own
  0-defect scope for this block.
- The full available test suite: 334 of 334 pass, unchanged from
  before this block's own audit -- confirming no regression, since no
  code was modified.
- No new regression tests were added: the same discipline already
  applied in `21-workflow` -- tests are added to prove a specific fix
  stays fixed, not manufactured for behavior directly verified and
  found correct.
