# `topocore.workflow` — Overview

## Scope

`Workflow` is the orchestrator for the whole engine: survey/LAS/LAZ
ingestion, ground/point classification, terrain construction (TIN,
DTM, contours), feature extraction (from both survey field codes and
point-cloud detection), CRS/vertical/georeferencing transforms, and
export to DXF/GeoPackage -- built entirely on top of modules already
closed elsewhere in this project (`survey`, `io`, `processing`,
`terrain`, `features`, `geodesy`, `dxf`, `gpkg`). This audit
deliberately did not re-audit any of those underlying modules'
own algorithms -- its own scope was orchestration: whether the
guarantees each module already provides actually survive passing
through `Workflow`.

## Real inventory — 8 files, 2391 lines

```text
topocore/workflow/
├── __init__.py       exports Workflow and its supporting types
├── artifacts.py         ArtifactStore, ArtifactRecord, ArtifactType (8 types)
├── exceptions.py          WorkflowError -> WorkflowStateError/
│                           StaleArtifactError/WorkflowValidationError/
│                           WorkflowExecutionError
├── history.py               ArtifactDependency, StageResult, WorkflowResult
│                            (append-only)
├── progress.py                ProgressEvent, ProgressObserver (a Protocol)
├── stages.py                   StageStatus, WorkflowStage (16 real stages)
├── validation.py                WorkflowValidator -- presence + transitive
│                                freshness checks
└── workflow.py                   Workflow itself -- 16 stage methods, all
                                    built on one shared `_execute_stage()`
```

`workflow.py` (1461 lines) is the largest single file audited across
this entire project -- larger than any file in `terrain`, `features`,
`gpkg`, or `dxf`.

## Architecture

```text
ArtifactStore          -- current version of each of 8 ArtifactType,
                           one record per type, never a dependency graph
WorkflowValidator       -- answers "is X present?" / "is X current?" --
                           never validates a stage's own parameters
_execute_stage()         -- the one place every stage's timing,
                            exception handling, ArtifactStore write, and
                            history append-only bookkeeping lives
16 WorkflowStage methods  -- each: validate preconditions -> gather
                             dependencies -> call _execute_stage(work)
```

Every one of the 16 stage methods delegates its actual execution
bracket to `_execute_stage()`, confirmed directly by reading each
one -- none inlines its own try/except around the domain work it
delegates to. `_execute_stage()` itself catches `Exception`
unconditionally (not a narrow tuple of specific types), wraps it into
`WorkflowExecutionError` with the original preserved via `from exc`,
and appends a `FAILED` `StageResult` to the append-only history
before re-raising -- confirmed by direct execution (see
[`validation.md`](./validation.md)) that this leaves no artifact
falsely marked valid and no corruption of prior, already-valid
artifacts.

## The 16 stages

```text
READ_SURVEY, READ_POINT_CLOUD                 -- mutually exclusive roots
CLASSIFY_GROUND, CLASSIFY_POINTS               -- 2 distinct algorithms
BUILD_TIN, BUILD_DTM, EXTRACT_CONTOURS          -- terrain construction
BUILD_FEATURES_FROM_SURVEY, DETECT_FEATURES      -- 2 distinct algorithms
RESOLVE_SIDES                                     -- optional, in-place
TRANSFORM_CRS, TRANSFORM_VERTICAL, GEOREFERENCE    -- optional, in-place,
                                                      each supports exactly
                                                      4 artifact types
EXPORT_DXF, EXPORT_GPKG                            -- lateral, never
                                                      written to the store
```

`ArtifactType` deliberately holds 8 types, not more: `GROUND_MASK` is
excluded by design (no real consumer; cheaper to derive a point count
than store the raw boolean array), and DXF/GPKG export reports are
lateral operations, never persisted to the store.

## Artifact flow and staleness

```text
POINT_CLOUD -> GROUND_CLOUD -> TIN -> DTM
     |                          |
     v                          v
 CLASSIFICATION_RESULT      EXTRACT_CONTOURS -> CONTOURS
     |
     v
 DETECT_FEATURES (optional: TIN, DTM, CLASSIFICATION_RESULT) -> FEATURE_COLLECTION
     |
SURVEY_POINT_SET -> BUILD_FEATURES_FROM_SURVEY -> FEATURE_COLLECTION
     |
     v
 RESOLVE_SIDES (in place) -> FEATURE_COLLECTION
     |
     v
 EXPORT_DXF / EXPORT_GPKG
```

`WorkflowValidator.require_current()` walks this graph transitively,
not just one hop back, via `WorkflowResult`'s own append-only history
-- confirmed with a real, executed reproduction of the exact scenario
its own docstring describes: `POINT_CLOUD` re-versioned (via
`transform_crs()`) after `GROUND_CLOUD`/`TIN`/`DTM` were built from
its earlier version correctly marks `DTM` stale, 2 hops removed, even
though `DTM`'s own immediate dependency (`TIN`) was never itself
rebuilt and so never changed its own recorded version number. See
[`validation.md`](./validation.md) for the complete real-execution
account.

## Historical defects -- already fixed, confirmed still fixed, not new findings of this audit

This package's own source documents 6 distinct, already-fixed
defects directly in code comments, each confirmed still correctly
fixed during this audit's own real-execution testing:

1. **PR19** -- `read_point_cloud()`'s format dispatch used to route
   every non-`.laz` file to `LASReader` regardless of its real
   format.
2. **PR20** -- `export_dxf()`/`export_gpkg()` used to pass
   `**exporter_kwargs` directly to the exporter's own constructor,
   which does not accept arbitrary keywords.
3. **`GroundManager.extract()`** used to drop the source cloud's own
   `crs`, discovered while adding `transform_crs()`'s own safety
   checks.
4. **PR20 coverage phase** -- `detect_features()` used to also track
   `GROUND_CLOUD`'s own staleness, even though
   `DetectionContext` has no field for it at all, causing a real,
   reproducible false-positive `StaleArtifactError`.
5. **`transform_crs()`'s own "already-transformed" safety check** --
   without it, transforming an artifact whose CRS already matches
   the transformer's own target would silently re-apply a second
   transformation.
6. **`georeference()`'s own "already has a CRS" safety check** --
   without it, running control-point georeferencing twice on the
   same artifact would silently double-transform it.

None of these required any further work during this audit -- they
are recorded here because they are real, valuable evidence this
package's own error paths have been exercised and hardened over
multiple prior sessions, not because this audit found anything new
to fix. See [`limitations.md`](./limitations.md) for why they are
not listed as current limitations.

## Result of this audit: 0 new defects, 0 code changes

Unlike `18-features` (3 defects fixed), `19-gpkg` (1 defect fixed),
and `20-dxf` (1 defect fixed), this audit found no functional defect
in `topocore.workflow` and made no code change to it. This is stated
plainly, not as a lower-effort audit: the same 5-point methodology
was applied, including targeted real-execution reproductions of the
highest-risk scenarios (transitive staleness, a malicious
`ProgressObserver`, a deliberately-failed stage's own downstream
recovery, consecutive `transform_crs()` calls, `export_gpkg()`'s own
EPSG resolution policy) -- see [`validation.md`](./validation.md) for
the complete account of what was actually executed, not merely read.

## Where to go next

- [`validation.md`](./validation.md) -- the complete real-execution
  evidence for Points 1-5, with no defect to report.
- [`limitations.md`](./limitations.md) -- pre-existing, out-of-scope
  findings only, plus the absence of a prior test suite for this
  package.
