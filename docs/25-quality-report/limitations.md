# Limitations

## No overall quality score

`QualityResult` never combines its sections into a single score, a
pass/fail verdict, or a red/yellow/green status. This is a deliberate
design decision (see [`overview.md`](./overview.md) and
[`validation.md`](./validation.md)'s own Point 2), not something
deferred by oversight: combining independent indicators into one
number requires weights and acceptance thresholds that are specific
to a project's own technical specification, which this module has no
way to know.

## No normative/regulatory thresholds

Nothing in this version compares a computed value (RMSE, density,
coverage, classification percentage) against any external standard,
specification, or regulation. Every number is reported as-is.

## RMSE alone never determines acceptance

`AccuracySection.rmse_total` (and its horizontal/vertical components)
is a computed statistic, not a judgment. Whether a given RMSE is
acceptable depends entirely on a project's own accuracy
specification -- this module does not encode, assume, or guess at
one.

## Classification may not be available at all

`ClassificationSection` is optional both in whether it's computed at
all and, within it, whether a `matched_code` was given.
`code_provided=False` produces `ground_percentage=None` -- confirmed
distinct from `0.0` -- specifically so "not measured" is never
confused with "measured as zero."

## Coverage and density are indicators, not absolute measures

`CoverageSection.coverage_percentage` depends entirely on the grid's
own `cell_size`, `bounds`, and `min_points_per_cell` -- the same
point data analyzed with a different `cell_size` produces a different
result. `DensitySection`'s own `points/m2` units are only meaningful
for a projected CRS; a geographic CRS produces a warning
(`DENSITY_GEOGRAPHIC_CRS`) but the value is still reported, not
suppressed or corrected.

## PDF output requires `reportlab`, an optional dependency

`QualityResult.to_json()` never requires it. `render_pdf()` raises
`PDFRendererUnavailableError` with a clear install message
(`pip install topocore[quality-report]`) if `reportlab` isn't
installed. As of this version, `reportlab` is not yet formally added
to `pyproject.toml`'s own `[project.optional-dependencies]` -- that
addition (and the corresponding `[[tool.mypy.overrides]]` entry) is
still pending, tracked as future work, not part of this closed
version.

## `mypy` on the renderer depends on the optional dependency's own stub availability

`topocore/quality/pdf_renderer.py` shows an `import-untyped` finding
for `reportlab.platypus` under `mypy` -- confirmed the same category
of finding already present elsewhere in this project for other
optional dependencies (`ezdxf`, `pyyaml`). Not a functional defect;
resolved by an override entry once `reportlab` is formally added as
an extra.

## No dashboard, no remote API

Explicitly out of scope for this version and not designed toward at
all -- no code hook, no placeholder, no partial implementation exists
for either. `QualityResult.to_json()` is intended as the shape a
future API could serve, but no API exists yet.

## `QualityAnalyzer` is not a stable API

Despite being functional, tested (23/23), and integrated (361/361 on
the full project suite), this is still a first version with a
deliberately simple contract. Signatures, field names, and even the
existence of specific indicators may change before any future,
broader version. Code depending on `topocore.quality` today should
expect that.

## No cross-section consistency is enforced

Each `analyze_*` method takes its own independent input array or
object. Nothing in `QualityAnalyzer` verifies that `density`,
`coverage`, and `classification` were computed from the *same*
underlying point cloud -- a caller could accidentally mix data from
different projects into a single `QualityResult` with no warning.
This was identified during the Point 2 contract discussion and
explicitly left unresolved for this version (see
[`validation.md`](./validation.md)) rather than adding a requirement
that would complicate the simple, independent-sections design this
version deliberately chose.
