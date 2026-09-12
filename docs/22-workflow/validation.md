# Validation — 0 defects found, complete real-execution evidence

## Methodology, identical to `16-analysis`/`16-terrain`/`18-features`/`19-gpkg`/`20-dxf`

```text
Point 1: real inventory       -> Point 2: contracts/consumers ->
Point 3: integration          -> Point 4: error paths         ->
Point 5: regression
```

Unlike the prior 4 blocks in this same PR22 pass, this methodology
did not lead to a defect here -- every real-execution reproduction
below confirmed the documented contract held exactly as stated. This
document exists to record that evidence, not to justify a fix.

## The core exception boundary — confirmed, by design and by execution

`_execute_stage()` catches `Exception` unconditionally, not a narrow
tuple of specific types -- this is the structural reason this audit
did not find the "an exception type not in the caught tuple silently
bypasses recovery" defect class found 3 times in `18-features`/
`19-gpkg`/`20-dxf`: there is no tuple here to have a gap in.

Two exception categories are kept deliberately, architecturally
distinct -- confirmed both by reading the code and by real execution,
not merely by the exceptions' own docstrings:

- **`WorkflowStateError`** (and its `StaleArtifactError` subtype):
  raised for precondition failures -- a missing or stale artifact, an
  unsupported `artifact_type`, an already-referenced artifact given
  to `georeference()`, an EPSG conflict in `export_gpkg()`. These
  checks run *before* `_execute_stage()` is ever called, so this
  exception type propagates directly, never wrapped into
  `WorkflowExecutionError`. Confirmed directly: calling `export_gpkg()`
  with an `epsg` that disagrees with the `FeatureCollection`'s own
  detected CRS raised `WorkflowStateError` directly, with no
  `WorkflowExecutionError` wrapping at all.
- **`WorkflowExecutionError`**: raised only for a failure *inside*
  `work()` -- the domain module's own real failure, always with the
  original exception attached via `__cause__`. Confirmed directly:
  calling `extract_contours(interval=-1.0)` on a real, valid `TIN`
  raised `WorkflowExecutionError` whose own `__cause__` was the
  original `TerrainValidationError` ("Contour interval must be
  greater than zero"), unaltered.

This is not treated as 2 inconsistent styles -- it is a deliberate
architectural boundary (validation vs. execution), stated explicitly
in `workflow.validation`'s own module docstring, and confirmed to
hold consistently across every stage method checked.

## The most important question: does a stage failure leave the `Workflow` in a coherent state?

Confirmed with a real, executed reproduction, not by inspection alone:

```text
1. Workflow: read_point_cloud() -> classify_ground() -> build_tin()
   (all 3 succeed; TIN v1 present)
2. extract_contours(interval=-1.0)  -- deliberately invalid
   -> WorkflowExecutionError raised, __cause__ = TerrainValidationError
3. Confirmed immediately afterward:
   - CONTOURS is NOT present in the store (has() == False) --
     no artifact was falsely marked valid
   - TIN is still present, still at v1, byte-identical to before
     the failed call -- the prior, valid artifact was not touched
   - history now has 4 entries; the 4th has status=FAILED,
     produced=None, and .error holding the original
     TerrainValidationError (not the wrapping WorkflowExecutionError)
4. Confirmed the Workflow remains fully usable afterward:
   - build_dtm(), an independent stage, ran successfully
   - extract_contours(interval=1.0) -- the SAME stage, retried with
     valid parameters -- also ran successfully, producing CONTOURS v1
```

This directly answers the central risk this audit's own Point 4 was
designed to rule out: a stage failure never leaves a falsely-valid
artifact, never corrupts a prior valid artifact, and never blocks a
legitimate subsequent call (whether a different stage or a retry of
the same one).

## `ProgressObserver` fault isolation — confirmed by execution, not only by docstring claim

`progress.py`'s own module docstring states plainly that no fault
isolation happens in that module itself -- "the try/except around
every call to `observer.on_progress(...)` lives entirely in
`workflow.py`." Confirmed directly:

- A deliberately broken `ProgressObserver` (`on_progress()` always
  raises `RuntimeError`) was passed to a real `Workflow`; a real
  `read_point_cloud()` call completed successfully regardless --
  confirmed by checking the resulting `POINT_CLOUD` artifact's own
  version. The `RuntimeError` was logged (`logger.warning`, with
  `exc_info=True`) but never propagated.
- For the deliberately-failed `extract_contours()` reproduction
  above, the observer received exactly one event
  (`"starting extract_contours"`) and never a false
  `"finished extract_contours"` -- confirmed by recording every
  message a real observer received during the failure.

## Transitive staleness detection — confirmed with the exact scenario its own docstring describes

`WorkflowValidator._is_stale()`'s own docstring gives a specific
example (`POINT_CLOUD v1 -> TIN v1 -> DTM v1`, then `POINT_CLOUD`
re-versioned without rebuilding `TIN`/`DTM`) as the reason the check
must be transitive, not a single-hop version comparison. Reproduced
directly, through the real public API (not by manipulating the store
by hand): `read_point_cloud()` -> `classify_ground()` -> `build_tin()`
-> `build_dtm()`, then `transform_crs(POINT_CLOUD, ...)` (a real
`CoordinateTransformer`, EPSG:4326 -> EPSG:32618, with realistic
geographic coordinates) to advance `POINT_CLOUD` to v2 without
rebuilding anything downstream. Confirmed:

- `wf.artifact(GROUND_CLOUD)` raised `StaleArtifactError` (1 hop).
- `wf.artifact(DTM)` also raised `StaleArtifactError`, 2 hops removed
  from the actually-changed artifact, even though `DTM`'s own
  immediate dependency (`TIN`) never itself changed version --
  exactly the case a naive, non-transitive check would miss.

Also confirmed: the self-referential dependency skip
(`_is_stale()`'s own handling of a stage that reads and re-produces
the SAME `ArtifactType`, e.g. `TRANSFORM_CRS`) does not cause a false
positive. 2 consecutive `transform_crs()` calls on `POINT_CLOUD`
(v1->v2->v3) both succeeded with no spurious `StaleArtifactError`
against the artifact's own immediately-prior version.

## `export_gpkg()`'s own EPSG resolution policy — confirmed for both real outcomes

- A `FeatureCollection` with `crs="EPSG:4326"` and no explicit `epsg`
  argument: confirmed the export succeeded, using the detected EPSG
  automatically.
- The same collection with an explicit, disagreeing `epsg=32618`
  argument: confirmed `WorkflowStateError` raised directly (not
  wrapped), naming both the detected and the given EPSG, never
  silently picking one.

## Regression

- `ruff`, with the real project's own `pyproject.toml`: 0 findings
  anywhere in `topocore/workflow/`.
- `mypy`, same real configuration: 0 findings in `topocore/workflow/`'s
  own 8 files, confirmed both via a plain run (which also surfaced 32
  findings in `topocore/processing/classification/*` -- transitively
  imported, already documented pre-existing debt from that package's
  own earlier audit, not attributable to `workflow`) and via an
  isolated `--follow-imports=silent` run against `workflow/`'s own 8
  files alone, which reported 0 issues.
- The full available test suite: 334 of 334 pass, unchanged from
  before this block's own audit -- confirming no regression, since no
  code was modified.
- No new regression tests were added for this block: per this
  project's own established discipline (avoid manufacturing test
  coverage for behavior that was never found broken -- the same
  principle already applied when this audit reused an existing test
  file rather than duplicating it during `20-dxf`), every scenario
  above was verified with real, direct execution during this audit,
  and none revealed a defect that would justify a dedicated
  regression test. See [`limitations.md`](./limitations.md) for the
  related, separate point about this package having no pre-existing
  test suite at all.
