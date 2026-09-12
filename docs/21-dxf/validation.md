# Validation — 1 defect found and fixed

## Methodology, identical to `16-analysis`/`16-terrain`/`18-features`/`19-gpkg`

```text
finding -> evidence -> classification -> fix -> behavior test -> regression
```

---

## `is_index_contour()` performed unchecked arithmetic on untrusted `Feature` data

**Finding**: `is_index_contour()` (`topocore.dxf.layers`) used
`elevation`/`base`/`interval` -- all 3 sourced from a `Feature`'s own
`attributes`/`metadata.extra` -- directly in arithmetic and a
`round()` call, with only a narrow, pre-existing check for
`interval <= 0`/`every < 1` (both already raising plain `ValueError`).

**Evidence**: confirmed directly, by exhaustively probing every
combination of invalid input: `interval<=0` and non-finite
`elevation`/`base`/`interval` each raised a raw `ValueError`; `base=inf`
specifically raised `OverflowError` (from the underlying `round()` call);
a non-numeric `interval` (e.g. a string) raised `TypeError`. A 4th,
distinct and more dangerous case was also confirmed: `interval=inf`
raised nothing at all and silently returned `True` -- misclassifying
a contour as an "index"/MAJOR contour, a wrong answer, not a crash.

None of `ValueError`/`OverflowError`/`TypeError` is caught by
`DXFExporter.export()`'s own `except (DXFGeometryError,
DXFExportError)` block. Confirmed reachable via the real, public
`Feature`/`FeatureMetadata` API, not merely by calling the private
function directly: `topocore.features.terrain.contours.ContourDetector`
(the one real detector that produces `CONTOUR` features) does validate
`interval`/`base` at its own construction time -- but this is a
guarantee that ONE particular detector's constructor enforces, not
one `Feature`/`FeatureMetadata` (the public model DXF actually
consumes) enforces at the type level. Confirmed directly:
`FeatureMetadata(extra={"interval": 0.0, "base": float("inf")})`
constructs without error. A `FeatureCollection` containing 1 such
malformed `CONTOUR` feature alongside 2 genuinely valid features,
exported with `strict=False`, produced **no `.dxf` file at all** --
confirmed by checking `Path.exists()` directly -- silently violating
`DXFExportOptions.strict`'s own documented "skip this feature, keep
going" contract.

**Classification**: a real, confirmed defect -- and, as established
during [`overview.md`](./overview.md)'s own investigation, the 3rd
confirmed instance of an already-documented, recurring failure shape
in this exact package (`layer_for()`'s own PR19 fix; `export_dxf()`'s
own PR20 fix). `DetectionError` was confirmed NOT the right exception
here, unlike the analogous fix in `18-features`: the data did not fail
validation on its way OUT of `features` -- `Feature`/`FeatureMetadata`
themselves never guaranteed this in the first place, and the failure
surfaces entirely within `dxf`'s own consumption of that data. This is
`dxf`'s own concern to enforce, on whatever `FeatureCollection` it is
given, regardless of which upstream package (or a hand-built
collection from any other caller) produced it.

**Fix**: `is_index_contour()` now validates `elevation`, `base`, and
`interval` explicitly at its own top -- each must be a real number
(`int`/`float`, excluding `bool`) and finite -- before any arithmetic,
raising `DXFExportError` directly. The pre-existing `interval <= 0`/
`every < 1` checks were changed from plain `ValueError` to the same
`DXFExportError`, for consistency (both are now the same class of
"this input is invalid" condition). Because `DXFExportError` is
already one of the 2 exception types `DXFExporter.export()`'s own
per-feature try/except catches, **no change to `exporter.py` was
needed at all** -- confirmed directly, by diffing the pre- and
post-fix source: `exporter.py` is byte-for-byte identical.

**Behavior test**: confirmed directly, with a real, queried `.dxf`
export, not merely a caught exception:

- All 9 invalid-input shapes (`interval` `<=0`/`NaN`/`Inf`/non-numeric;
  `base` `NaN`/`Inf`; `elevation` `NaN`/`Inf`) now raise `DXFExportError`
  from `is_index_contour()` directly.
- A collection with 2 valid features + 1 malformed `CONTOUR`, under
  `strict=False`: the resulting `.dxf` file now exists;
  `feature_count=3`, `skipped_features=1`, with a warning naming the
  specific validation failure.
- The identical collection under `strict=True`: still aborts (now
  with `DXFExportError` specifically, matching this fix's own
  exception type -- not `DXFValidationError`, which is a distinct,
  separate path for issues `DXFValidator.validate()` itself catches),
  and no file is left at the target path.
- A fully valid `CONTOUR` feature alongside a valid feature: confirmed
  to export identically to before the fix (`feature_count=2`,
  `skipped_features=0`).
- `is_index_contour()` on genuinely valid input: confirmed unchanged
  (`True`/`False` as expected for an index vs. non-index elevation).

**Regression**:

- `exporter.py`, `entities.py`, `mapping.py`, `xdata.py`,
  `validation.py` -- confirmed completely untouched (only `layers.py`
  was modified).
- `ruff`/`mypy`, with the real project's own `pyproject.toml`: clean
  on `layers.py`.
- 13 new regression tests, covering every invalid-input shape, the
  real end-to-end `strict=False`/`strict=True` behavior, and a fully
  valid collection -- all pass.
- The full available test suite: 334 of 334 pass.

---

## The other 2 candidates investigated in this same audit -- confirmed NOT defects

`write_entity()`'s own `else: raise NotImplementedError(...)` and
`XDataEncoder.encode()`'s own `TypeError` were both investigated with
the same reachability standard applied to `is_index_contour()` above,
and both confirmed currently unreachable via any legitimately-
constructed `Feature`:

- `GeometryType`'s 4 real members map, one-to-one and exhaustively,
  onto `GeometryMapper.decide()`'s 4 possible `DXFRepresentation`
  outputs -- confirmed directly, no 5th case exists today.
- `build_feature_xdata()`'s own pre-filtering
  (`is_xdata_encodable()`) confirmed, with a real `Feature` carrying a
  nested-dict attribute, to already remove anything `XDataEncoder.encode()`
  would otherwise reject, before it is ever called.

Neither was modified. This distinction -- a bare `raise` statement is
not, by itself, evidence of a defect; reachability from the real
public API must be demonstrated -- is what separated these 2 from the
1 real, fixed defect above. See [`limitations.md`](./limitations.md)
for how these 2 are recorded going forward.

## What this fix does not cover

No test file for `topocore.dxf` existed anywhere available to this
audit, the same situation already noted for `18-features`/`19-gpkg`.
The 13 tests added here are, as far as this audit could confirm, the
first tests this package has had.
