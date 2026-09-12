# Quality Report — Overview

## What this is, and what it is not

This is the first implementation of a new functional line for
TopoCore, built on top of already-existing, already-audited
components. It is **not** part of PR22, and it is **not** yet a
stable, public TopoCore API. It is a closed first version (Points
1-6 of its own dedicated audit, complete), with a deliberately simple
V1 contract.

```text
QUALITY REPORT
════════════════════════════════════════
Functional implementation       CLOSED
Dedicated tests                 23/23
Global suite                    361/361
Ruff                            clean
MyPy                            reportlab optional, override pending

Contract                        SIMPLE / V1
Dashboard                       OUT OF SCOPE
Remote API                      OUT OF SCOPE
Normative thresholds            OUT OF SCOPE
```

## Purpose

A structured, computed answer to *"how trustworthy is the result I
just produced?"* -- positional accuracy against control, point
density, spatial coverage, and classification distribution -- as a
first-class capability, not something every user has to reassemble
by hand from 3-4 separate TopoCore modules.

## Architecture

```text
                    TopoCore
                       |
              +--------v--------+
              | QualityAnalyzer |
              +--------+--------+
                       |
                QualityResult
                       |
              +--------+--------+
              v                 v
            JSON               PDF
       (no reportlab       (reportlab,
        required)           optional extra)
```

3 modules, confirmed by real execution to be genuinely independent:

- **`topocore.quality.models`** -- `QualityResult` and its sections
  (`AccuracySection`, `DensitySection`, `CoverageSection`,
  `ClassificationSection`, `SpatialReference`, `Diagnostic`). Plain,
  frozen dataclasses. Zero imports of any presentation library.
- **`topocore.quality.analyzer`** -- `QualityAnalyzer`, which
  computes each section. Reuses 2 already-audited TopoCore
  components directly (`RMSEAnalysis.compute()`,
  `StatisticsAnalysis().density()`); the rest (coverage,
  classification, CRS handling) is new to this module.
- **`topocore.quality.pdf_renderer`** -- the *only* module that
  imports `reportlab`. Takes an already-computed `QualityResult` and
  renders it; never recomputes anything.

## The indicators, as of this version

```python
from topocore.quality.analyzer import QualityAnalyzer

analyzer = QualityAnalyzer(project_name="...")

analyzer.analyze_accuracy(reference, observed)           # -> AccuracySection | None
analyzer.analyze_density(points_xy, resolution=1.0, crs=None)  # -> DensitySection | None
analyzer.analyze_coverage(points_xy, bounds, cell_size=..., min_points_per_cell=1)  # -> CoverageSection | None
analyzer.analyze_classification(codes, matched_code=None)  # -> ClassificationSection | None

result = analyzer.build(accuracy=..., density=..., coverage=..., classification=..., crs=...)
```

Every `analyze_*` method is independent and optional -- call only the
ones your data supports. None of them fabricate a value when the
input is insufficient; they return `None` and record a
`Diagnostic(code, severity, message)` on the analyzer instead.

## Integration with TopoCore

Confirmed with real execution, not merely described:

- **`accuracy`**: `reference` is known control-point truth (surveyed
  independently); `observed` is typically TopoCore's own `TIN`/`DTM`
  interpolated at those same XY locations (`TIN.interpolate(x, y)`
  -- see [`terrain.md`](../24-usage/terrain.md)). `QualityAnalyzer`
  never calls `TIN.interpolate()` itself -- building `observed` is
  the caller's own responsibility, before calling
  `analyze_accuracy()`.
- **`density`/`coverage`**: take raw XY arrays, typically extracted
  from a `PointCloud`'s own `Chunk` (see
  [`primitives.md`](../24-usage/primitives.md)) -- usually
  `GROUND_CLOUD` (post `classify_ground()`), since ground density is
  what's usually relevant.
- **`classification`**: takes `ClassificationResult.labels` directly
  -- confirmed these are real ASPRS integer codes, TopoCore's own
  native convention from `classify_points()`, not an external
  assumption this module introduces.
- **`spatial_reference`**: `PointCloud.crs`/`FeatureCollection.crs`,
  passed straight through as a string -- registered, never validated
  in this version.

## The optional `reportlab` dependency

Confirmed by real execution: `QualityResult.to_json()` works
completely independently of whether `reportlab` is installed.
`render_pdf()` is the only function that needs it, and raises a clear
`PDFRendererUnavailableError` (naming the future
`pip install topocore[quality-report]` extra) if it's missing --
`pip install topocore` itself would never need to pull it in.

## What "quality" does and does not mean here

Every indicator is independent and purely descriptive. `QualityResult`
does **not** compute, and this version does not attempt to compute,
an aggregate quality score, a pass/fail verdict, or a red/yellow/green
status. An RMSE of 0.045m is reported as exactly that -- a number --
never as "good" or "bad": whether that's acceptable depends on a
project's own specification, which this module does not know and
does not guess at. See [`limitations.md`](./limitations.md) for the
complete, explicit list of what each indicator does not claim.

## Where to go next

- [`validation.md`](./validation.md) -- the complete real-execution
  evidence: the contract (Point 2), the integration mapping (Point
  3), all 14 error-path cases (Point 4), and the 2 real regressions
  found and fixed during that review.
- [`limitations.md`](./limitations.md) -- what this version
  deliberately does not do.
