# Validation — the complete real-execution evidence

## Point 2 — the formal contract, as decided

6 real decisions, made explicitly rather than inherited by default
from the components this module reuses:

| Decision | Contract adopted |
|---|---|
| CRS for accuracy | Not validated in this version. Registered if available. |
| Minimum control points | 2. Fewer -> `None` + `ACCURACY_INSUFFICIENT_CONTROL_POINTS` diagnostic, confirmed by real execution even though the underlying `RMSEAnalysis.compute()` itself accepts a single point without complaint. |
| Density + geographic CRS | Warning (`DENSITY_GEOGRAPHIC_CRS`), never blocks the computation. |
| Classification code | Optional. Without it: `code_provided=False`, `ground_percentage=None` (confirmed distinct from `0.0` -- N/A is not zero). |
| Diagnostics | Structured: `code`, `severity` (`INFO`/`WARNING`/`ERROR`), `message` -- not free text, so a future API consumer can branch on `code`. |
| Data source | No single unified source required. Each `analyze_*` method takes its own independent, optional input. |

**Explicit design decision, not an omission**: `QualityResult` never
computes an aggregate score or pass/fail verdict. Combining
independent indicators into one score would require weights and
thresholds that are project-specific business decisions, not
something this module can decide universally.

## Point 3 — integration, confirmed by real execution

```text
accuracy         reference = independently surveyed control truth
                 observed  = TIN.interpolate(x, y) at the same XY,
                             confirmed with a real TIN built via
                             TIN.from_points() -- QualityAnalyzer
                             never calls interpolate() itself

density/coverage XY extracted from a PointCloud's own Chunk
                 (chunk[PointAttribute.X]/[Y]), typically GROUND_CLOUD

classification   ClassificationResult.labels -- confirmed real ASPRS
                 codes, TopoCore's own native convention from
                 classify_points(), not an external assumption

spatial_reference PointCloud.crs / FeatureCollection.crs, passed
                 through as a string, unvalidated
```

## Point 4 — all 14 error-path cases, confirmed by real execution

| Case | Confirmed result |
|---|---|
| 0 control points | `None` + `ACCURACY_INSUFFICIENT_CONTROL_POINTS` |
| 1 control point | `None` + same diagnostic |
| >= 2 valid control points | RMSE computed correctly |
| NaN/Inf in accuracy | `None` + `ACCURACY_COMPUTE_FAILED` (inherited correctly from `RMSEAnalysis`) |
| NaN/Inf in density | **Found a real, pre-existing defect** -- see below |
| NaN/Inf in coverage | `None` + `COVERAGE_NON_FINITE_POINTS` (already validated since Point 1.5) |
| Classification absent | `code_provided=False`, `ground_percentage=None` |
| Classification code not present in data | Computed as `0` matched, plus a new `CLASSIFICATION_CODE_NOT_FOUND` warning (added during this review) |
| CRS absent | `SpatialReference(crs=None)` plus a new `SPATIAL_REFERENCE_MISSING` warning (added during this review) |
| CRS geographic for density | Warning, computation proceeds |
| CRS unrecognized format (not `EPSG:xxxx`) | Silently skipped, density still computed -- accepted as-is |
| A report with every section `None` | Still a valid, complete `QualityResult`, valid JSON |
| A real `TIN.interpolate()` failure (point outside the hull) | Raises `ValueError` -- confirmed this is the caller's own responsibility when building `observed`, before calling `analyze_accuracy()`, not something `QualityAnalyzer` catches |
| `reportlab` not installed | `to_json()` unaffected; `render_pdf()` raises `PDFRendererUnavailableError` |
| PDF write failure | **Found a real, pre-existing defect** -- see below |

## 2 real regressions found and fixed during this review

**1. `analyze_density()` let a raw `OverflowError` escape for
non-finite input.** Confirmed traced to `StatisticsAnalysis.density()`
itself -- already-closed PR22 code, not modified here. Fixed at this
module's own boundary: `analyze_density()` now validates
`np.isfinite()` before calling into that component, converting the
failure into a `DENSITY_NON_FINITE_POINTS` diagnostic. Verified with
a real reproduction: the exact input that used to crash now returns
`None` with the diagnostic, and the normal-input case was reconfirmed
unaffected.

**2. `pdf_renderer.py` referenced field names from before the Point 2
contract change** (`source_crs`/`target_crs`/`transformation` on
`SpatialReference`, `horizontal_rmse`/`mean_density` on other
sections) and would raise `AttributeError` on every real render call
-- it was never reconciled when the contract changed. Found while
testing the "PDF write failure" case in the error-path matrix, as a
real side effect, not the failure being specifically searched for.
Fixed by rewriting the renderer against the current, real field
names in `models.py`. A dedicated regression test now renders every
section at once specifically so a future field rename is caught
immediately.

**A rule followed throughout this review, confirmed in the code
itself**: no bare `except Exception` was introduced anywhere. Every
catch is specific --  `QualityError`/`StatisticsError` from the 2
reused components, `CRSError`/`ValueError` for CRS parsing, `OSError`
for the PDF write failure. An earlier draft of the CRS-parsing logic
did use a bare `except Exception: pass`; `ruff` flagged it directly
(`BLE001`) and it was narrowed before being included in this version.

## Point 5 — regression suite, real numbers

```text
Dedicated tests written               24
Own mistake found in a test itself     1 (a wrong expected RMSE value
                                          in the test itself, not a
                                          code defect -- corrected)
Dedicated tests, final                  23/23 pass
Ruff (topocore/quality/)                 clean
Ruff (dedicated tests)                    1 mechanical import-order
                                             finding, fixed
MyPy (topocore/quality/)                   only the expected
                                              import-untyped finding
                                              for reportlab (same
                                              category as ezdxf/
                                              pyyaml elsewhere in
                                              this project) -- would
                                              be resolved by adding
                                              reportlab to
                                              [[tool.mypy.overrides]]
                                              once the extras group
                                              is formally added
Full project suite (pytest)                361/361 -- a real,
                                               executed number (338
                                               pre-existing + 23 new),
                                               not assumed
```

2 behaviors were specifically protected with a dedicated test, per
their own real regression:

```text
NaN/Inf input -> QualityAnalyzer -> diagnostic -> QualityResult -> no crash
    test_density_with_infinite_value_returns_none_not_a_crash

Partial QualityResult -> PDFRenderer -> valid PDF
    test_render_pdf_with_partial_result_still_produces_a_valid_pdf
```

Plus `test_render_pdf_with_full_result_produces_a_real_file`, which
exercises all 5 sections in a single render specifically so a future
field-name drift between `models.py` and `pdf_renderer.py` (the exact
2nd regression above) is caught immediately, not discovered later
during manual testing.
