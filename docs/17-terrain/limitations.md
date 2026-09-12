# Limitations

## Pre-existing lint findings — confirmed under the real project configuration, deliberately not touched by this audit's own corrections

Confirmed 3 times over: before this session's environment reset,
after reconstruction with default/manual flags, and finally with the
real project's own `pyproject.toml`, auto-discovered from the
package root -- the exact same 19 `ruff` findings, in the exact same
files this audit never modified. Left unaddressed deliberately, to
keep this audit's own scope to the 9 agreed items in
[`validation.md`](./validation.md) -- the same precedent already
established during the companion `16-analysis` audit (`UP040`, 10
findings, same treatment).

**`mypy` is no longer part of this section.** The package's own one
real `mypy` finding (`nearest.py:91`, `no-any-return`) was
investigated with its own dedicated mini-audit -- confirmed to be a
pure `numpy` type-stub precision gap (`np.argmin(..., axis=...)`
inferring `Any` under the installed `numpy` 2.4.4), not a runtime
defect or a deliberate contract, and confirmed to have zero internal
consumers that a fix could affect -- and fixed as this audit's own
9th correction (a type-only local-variable annotation, no logic
change). `mypy`, with the real project's own configuration, now
reports **0 errors anywhere in `terrain/`** (32 files). See
[`validation.md`](./validation.md) for the complete account.

### `RUF022` / `RUF023` (`__all__`/`__slots__` not alphabetically sorted) — 11 findings

```text
_geometry.py:102, algorithms/__init__.py:26, base.py:140,
breaklines.py:146 (__slots__), constants.py:70, filters.py:165 (__slots__),
filters.py:303 (__slots__), filters.py:343, hillshade.py:108 (__slots__),
hillshade.py:206, models.py:177, nodata.py:158, types.py:54
```

Confirmed, by reading each one directly, to follow the same
deliberate semantic/logical grouping already established and
preserved (with a scoped, justified `noqa`) in the files this audit
did modify (`terrain/__init__.py`, `validation.py`) and in
`analysis/__init__.py` during the companion audit -- e.g.
`filters.py`'s own `__all__` groups `LaplacianSmoother` with
`laplacian_smooth` together, then `SpikeDetector` with
`detect_spikes`/`remove_spikes`, not alphabetically. Not flattened.

### `RUF046` (redundant `int()` cast on an already-integer value) — 4 findings

```text
models.py:147, models.py:154, sampling.py:177, sampling.py:178
```

Mechanical, zero-risk findings, the same category already cleaned up
project-wide during `16-analysis`'s own Point 1 lint pass -- but not
applied here, since doing so was outside this specific 9-item
correction batch's own agreed scope.

### `RUF007` (prefer `itertools.pairwise()` over `zip()`) — 1 finding

```text
breaklines.py:97
```

A style suggestion, not a behavior concern -- the current `zip(indices[:-1],
indices[1:])` pattern is correct as written.

### `I001` (import block formatting) — 1 finding

```text
algorithms/constrained_delaunay.py:35
```

## Declared but never connected — the same category as `LocalCRS`/`TransformationAccuracy` in `14-geodesy`

Confirmed by search across the entire codebase: each item below is
declared, and in most cases exported at package level, but has zero
real consumer anywhere.

| Item | Where | Status |
|---|---|---|
| `GridError` | `exceptions.py` | Declared, NOT exported, never raised anywhere |
| `DEFAULT_RESOLUTION`, `MIN_RESOLUTION`, `MIN_CONTOUR_INTERVAL`, `MAX_TRIANGLE_EDGE_LENGTH`, `MIN_SLOPE`, `MAX_SLOPE`, `MIN_ASPECT`, `MAX_ASPECT` (8 of 12 constants) | `constants.py` | Declared, not exported, zero real usage |
| `DEFAULT_NO_DATA` (9th unused constant) | `constants.py` | Declared, not exported; only ever referenced in one unrelated docstring comment, never as a real value |
| `AspectReference`, `ContourSmoothing`, `SlopeMethod`, `TriangulationMethod` (4 of 6 enums) | `enums.py` | Declared, not exported, zero real usage anywhere, not even in a comment -- likely representing planned, never-wired-up configurability (alternative slope/aspect-reference/contour-smoothing/triangulation algorithms) |
| `CellSize`, `Area`, `Volume`, `Distance`, `Coordinate` (5 of 10 type aliases) | `types.py` | Declared, not exported at package level |
| `GridDefinition` | `models.py` | Exported, but confirmed -- by `base.py`'s own docstring directly, not just by this audit -- to be "an unrelated, older grid model that `DTM` never actually used" |
| `validate_grid_definition()` | `validation.py` | Exported, its only real consumer is `GridDefinition` above; zero callers beyond its own declaration and tests |
| `validate_tin()` | `validation.py` | Exported, zero callers anywhere beyond its own declaration and tests |
| `nodata.py`'s 7 functions (`is_nodata`, `valid_mask`, `nodata_mask`, `valid_count`, `nodata_count`, `replace_nodata`, `fill_nodata`) plus `DEFAULT_NODATA` | `nodata.py` | All correct and tested in isolation, but zero real callers anywhere beyond their own tests -- `Raster.nodata` is the one place this convention is actually enforced in working code |

None of these were touched by this audit's own 9 corrections --
each is either unreachable dead code (safe to leave, no behavior
risk either way) or, for `GridDefinition`/`validate_grid_definition()`,
already self-documented by the project's own source as known,
intentional legacy.

## `BaseTIN` — a declared contract with zero real implementers

Covered in full in [`contracts.md`](./contracts.md): `BaseTIN.__subclasses__()`
is empty -- confirmed, `TIN` (the sole TIN implementation anywhere in
this codebase) does not inherit from it, despite `BaseTIN`'s own
docstring stating it is the interface "implemented by every TIN
model." Not fixed during this audit: doing so would mean either
renaming several of `TIN`'s own real, widely-used attributes, or
reworking `BaseTIN` to match `TIN`'s real shape -- a structural
change well beyond this batch's own agreed scope. Flagged here as a
confirmed, real finding for a future, deliberate decision.

## A theoretical, unpursued inconsistency in `remove_spikes()`

Covered in [`filters-breaklines.md`](./filters-breaklines.md):
`remove_spikes()` raises its own `TerrainError` when fewer than 3
vertices would survive spike removal, but if exactly the *surviving*
vertices happened to be collinear (a different failure mode), the
underlying `TIN.from_points()` call would raise `TriangulationError`
unwrapped instead. Not pursued: constructing this exact scenario
requires an initial point cloud close enough to already-degenerate
that it would typically fail earlier construction anyway, making it
a low-likelihood, mostly theoretical edge case rather than a
practical one.

## What this audit did not attempt

Comprehensive test coverage for `topocore.terrain` as a whole was
never the goal -- the real project's own test suite (22 files, 304
tests, confirmed run in full against the corrected tree) already
provides that, far more completely than anything this audit would
have added. This audit's own contribution was the 9 confirmed,
fixed defects in [`validation.md`](./validation.md), plus the
complete, honest inventory of pre-existing gaps recorded on this
page -- not a rewrite, and not a claim that `terrain` is now
"perfect," only that it is more precisely known than before.
