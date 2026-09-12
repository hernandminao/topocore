# Limitations

## `register_many()`'s own incremental (non-atomic) behavior — a documented operational consideration, not a defect

`FeatureCodeRegistry.register_many()` registers each definition in
sequence and stops at the first collision with an already-registered,
differently-defined code -- entries registered before that point
remain registered; entries after it are never attempted. Confirmed
directly, with a real reproduction against the actual built-in
default registry (see [`validation.md`](./validation.md) for the
complete scenario): loading a valid, 3-entry external catalog where 1
entry collides with an existing code leaves the 2 non-conflicting
entries registered, while the colliding code's own original
definition is confirmed unchanged.

**Why this is recorded as a limitation, not a defect**: no
documentation or docstring anywhere in this codebase promises
all-or-nothing semantics for this method, and the partiality is
confirmed benign in every real-execution reproduction performed --
no pre-existing definition is ever overwritten (the same identity
check that makes re-registering an identical definition idempotent
also protects against a colliding attempt silently replacing it),
the registry remains fully usable immediately afterward, and a
caller can retry cleanly with the non-conflicting subset. The only 2
real call sites for this method anywhere in this codebase
(`FeatureCodeRegistry.__init__()`'s own convenience wrapper, and
`default()`'s own call with the already-validated, 0-collision
`ALL_CODES`) never exercise this path in practice.

**Operational consideration for anyone layering an external catalog
onto an existing registry** (the pattern `from_json()`'s own
docstring explicitly recommends): a caller who catches the
`ValueError` from a colliding `register_many()` call should not
assume nothing was registered -- entries preceding the collision in
iteration order are already applied. If a caller specifically needs
all-or-nothing semantics, they should pre-check for collisions
themselves (e.g. via `get()` on each code before calling
`register_many()`) rather than relying on the exception alone to mean
"no changes were made."

## Confirmed pre-existing, out of this audit's own 0-defect scope

`ruff`, with the real project's own `pyproject.toml`: 0 findings
anywhere in `topocore/features/catalogs/` -- nothing to document
here, unlike every prior block in this PR22 pass.

`mypy`, same real configuration, isolated to this submodule's own 19
files: 1 finding -- `loaders/yaml_loader.py`'s own `_require_yaml()`,
`no-any-return` on `return yaml`. Confirmed the same category of
finding already documented multiple times elsewhere in this project
(`16-terrain`'s `nearest.py` before its own fix, `19-gpkg`'s
`geometry.py`) -- a third-party library's own imprecise or
unavailable type stubs causing a locally-`Any`-typed value to flow
into a function with a precise return annotation, not a runtime
defect. Not corrected here, consistent with this block's own 0-defect
scope.

## No test suite was available for this submodule

Confirmed directly: no `tests/features/catalogs/` (or equivalent)
directory was present in the reconstructed repository export this
audit worked from, and none was otherwise provided -- the same
situation already noted for `18-features`/`19-gpkg`/`20-dxf`/
`21-workflow`. Unlike `18-features`/`19-gpkg`/`20-dxf`, no regression
test was added here either, since no defect was found to attach one
to -- matching `21-workflow`'s own precedent. This submodule's own
`catalog_audit.run_audit()`, however, already functions as a
substantial, real, existing form of self-verification (confirmed
directly: 160 codes, 0 violations) -- this is not equivalent to a
maintained pytest suite, but it is real, existing coverage this audit
did not have to add.

## What this audit did not attempt

The 9 individual domain catalog files (`cadastre.py`, `control.py`,
`default.py`, `drainage.py`, `structures.py`, `terrain.py`,
`transportation.py`, `utilities.py`, `vegetation.py` -- 160 codes
combined) were not reviewed code-by-code for the semantic
correctness of each individual field-code-to-`FeatureType` mapping
(e.g., whether a specific survey abbreviation is assigned to
precisely the right `FeatureType`/`category`/`layer` a real surveying
convention would expect). This audit instead relied on
`catalog_audit.run_audit()`'s own real, executed, passing result
(160 codes, 0 violations) for structural/invariant correctness, and
verified the surrounding machinery (loaders, registry, validation
boundary, integration with `features`/`workflow`) directly. A defect
in one specific code's own semantic assignment, if one exists, would
not necessarily be caught by either this audit or `run_audit()`
itself, since neither checks a code's real-world surveying
correctness -- only its own internal structural consistency.
