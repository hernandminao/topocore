# Quality Engine — architectural vision (NOT current, stable functionality)

> **This page is different from every other page in `24-usage`.**
> Everything else in this guide documents real, existing, stable
> TopoCore syntax, verified against the actual codebase. This page
> does not. It records an **architectural decision and a validated
> prototype** for a capability that does **not** exist as part of
> TopoCore today, is **not** part of PR22, and is **not** a stable
> API. Nothing described here should be called from production code.
> If you came here looking for how to use TopoCore, see
> [`overview.md`](./overview.md) instead.

## Status, as decided

```text
PR22
════════════════════════════════════════
Functional packages                CLOSED
QA / Validation                    CLOSED (as a process)
Optimization                       CLOSED (as a process)
Usage / syntax (24-usage)          IN DOCUMENTATION

Quality Engine                     OUT OF PR22, entirely
  Architecture                     VALIDATED (real execution)
  Prototype                        VALIDATED (real execution)
  Formal contract                  NOT DONE -- future PR
  Production implementation        NOT DONE -- future PR
  Dashboard / web                  DISCARDED for now
```

`QualityAnalyzer`/`QualityResult`, as prototyped, are **not** a
stable TopoCore API. Their real signatures, error contracts, and
even their existence in a future release may change completely
before any formal PR. Nothing here is a promise.

## Why this line of work might be worth pursuing later

TopoCore's own value proposition doesn't have to stop at "processes
and exports point clouds." A structured answer to *"how trustworthy
is the result I just produced?"* -- CRS verified, RMSE against
control, density, classification distribution, coverage, geometric
integrity -- has real value for topography, LiDAR, photogrammetry,
civil works, and public-sector deliverables, and it composes
naturally with what TopoCore already has (CRS, classification,
features, TIN/DTM, analysis, cut/fill).

The bet worth recording: the differentiated capability is
`QualityAnalyzer` + `QualityResult` as a structured model consumable
by anything -- not a PDF generator. A PDF is one possible rendering
of that model, never the goal itself.

## The architectural boundary, validated with real execution

```text
                    TOPOCORE (hypothetical, future)
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
        (future API)      (optional extra)

Dashboard / Web          DISCARDED for now -- not designed, not
                           referenced, no code hook left for it
```

Confirmed by real execution during this exploration (not merely
argued):

- `QualityAnalyzer.analyze_*()` methods reuse real, already-audited
  TopoCore components directly (`RMSEAnalysis.compute()`,
  `StatisticsAnalysis().density()`) where they apply, rather than
  reimplementing them.
- `QualityResult` (and its nested sections --
  `PositionalAccuracy`/`DensitySection`/`CoverageSection`/
  `ClassificationSection`/`SpatialReference`) is a plain, frozen
  dataclass model with **zero** import of any presentation library.
- `QualityResult.to_json()` works completely independently of
  whether `reportlab` is even installed -- confirmed by real
  execution.
- A separate `render_pdf(result, path)` function is the **only**
  place that imports `reportlab`, with its own clear
  `PDFRendererUnavailableError` if it's missing, naming the future
  optional install extra (`pip install topocore[quality-report]`).
  Confirmed: `pip install topocore` itself would never need to pull
  in `reportlab` under this design.
- A deliberately-broken input (mismatched array shapes) was confirmed
  to produce a `None` section plus a readable diagnostic string,
  never a crash -- one bad section does not abort the rest of a
  `QualityResult`.

## Explicit boundaries agreed for this vision

- `QualityAnalyzer` -- calculation only.
- `QualityResult` -- an independent result model, no presentation
  concerns.
- JSON -- structured output, the intended shape for a future API.
- PDF -- presentation, behind an optional dependency
  (`quality-report` extra), never mandatory.
- Dashboard / web -- explicitly out of scope. Discarded for now, not
  designed, no code written toward it.
- A formal contract (units, required vs. optional fields, exact
  definition of every indicator, CRS handling, error paths) --
  deferred to a future PR, not decided here.
- A production-ready implementation -- deferred to a future PR.
- This prototype is not, and should not be presented as, a closed or
  official TopoCore feature.

## A real, open question this vision does not resolve

`ground_percentage`/`matched_percentage` and `coverage_percentage`
are descriptive indicators, not proof of correctness -- a high
ground percentage doesn't mean the classification is *accurate*, and
a coverage result depends entirely on the grid's own `cell_size` and
`min_points_per_cell` choices. Any future formal contract for this
capability needs to resolve, explicitly, what each indicator is
allowed to claim and what it explicitly does not certify -- this
vision intentionally leaves that undecided rather than picking an
answer prematurely.

## A broader shape this could grow into, unbuilt and unvalidated

For reference only -- none of this is designed, scoped, or committed:

```text
QualityResult
|-- integrity        (duplicates, NaN/Inf, outliers)
|-- geometry          (TIN validity, degenerate geometry)
|-- density
|-- classification
|-- accuracy          (RMSE against control)
|-- terrain            (DTM coverage, cut/fill consistency)
|-- coverage
`-- diagnostics         (processing traceability)
```

This table is a direction to consider in a future PR's own Point 0
(scope decision), not a specification.
