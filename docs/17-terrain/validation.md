# Validation — the 9 defects found and fixed

## Methodology

Every defect below followed the same sequence, in order, and none
was "fixed" before all 4 earlier steps were satisfied:

```text
finding -> evidence -> classification -> fix -> behavior test -> regression
```

"Evidence" means direct execution against the real source, not
inference from a docstring or a variable name. "Classification"
means checking, before writing any fix, whether the real project's
own test suite already treats the observed behavior as a deliberate
contract -- exactly the discipline this session adopted after a
real, corrected mistake during the companion `16-analysis` audit
(a fix there was applied, then fully reverted, once the real
project's own test suite turned out to explicitly assert the
"defect" as its own intended behavior). None of the 9 defects below
had any such counter-evidence: each was confirmed, by search, to
have zero test anywhere asserting the old behavior as intentional.

**A note on this document's own evidence chain**: this session's own
sandbox lost its working copy mid-audit (an environment reset, not a
code issue). All 9 fixes below were re-applied from scratch against
a freshly reconstructed copy of the real repository (a full
`repo-to-text` export), and re-verified with real execution --
including, for the first time in this audit, this project's own
complete real test suite (22 files, 304 tests, not a partial sample)
run directly against the corrected tree. Every "confirmed" claim
below reflects that final, complete re-verification, not the
pre-reset run.

---

## 1. `Grid.point()` — permanently broken, removed

**Finding**: `Grid.point()`'s own docstring promises
`z=float("nan")`, but constructs a `Point3D`.

**Evidence**: `Point3D.__post_init__` calls `validate_coordinate()`
on all 3 coordinates, which rejects `NaN` unconditionally. Calling
`Grid.point()` at all -- for any row/column, any grid -- always
raised `MathError`. The real project's own coverage report showed
this method's own body as never executed by any test; confirmed, by
search, zero callers anywhere in the codebase or its test suite.

**Classification**: a real, confirmed defect, not a design choice.
No test anywhere treats a raised `MathError` here as intended
behavior -- the method simply never succeeded.

**Fix**: removed. `Grid.cell()` already provides the equivalent
"point at this grid location, elevation not yet known" need
correctly, using `Cell` (which, unlike `Point3D`, freely accepts
`NaN`) -- confirmed still working, unaffected. The now-unused
`Point3D` import was removed from `grid.py` alongside it.

**Behavior test**: confirmed, by search across the reconstructed
real repository, zero remaining references to `Grid.point()`
anywhere.

**Regression**: `ruff`/`mypy` clean on `grid.py`; all 304 real
project tests pass, including every `test_grid_bounds.py` case.

---

## 2. `ConversionError` — declared, never exported

**Finding**: `terrain.exceptions.ConversionError` exists, is raised
by `conversion.pointcloud_to_points()` for an empty `PointCloud`, and
is even included in `exceptions.py`'s own `__all__` -- but was never
imported or re-exported by `terrain/__init__.py`, unlike every one
of its 6 sibling exceptions.

**Evidence**: `from topocore.terrain import ConversionError` raised
`ImportError` before the fix; `from topocore.terrain.exceptions import
ConversionError` worked. Confirmed the exception is genuinely
raised in real code (`conversion.py`), not dead.

**Classification**: a real gap, not a design choice -- every other
exception in this package's own domain hierarchy is exported at
package level; there was no comment or pattern suggesting
`ConversionError` was deliberately withheld.

**Fix**: added to `terrain/__init__.py`'s own `.exceptions` import
and `__all__`.

**Behavior test**: `from topocore.terrain import ConversionError`
confirmed to work; confirmed to be the identical class object as
`topocore.terrain.exceptions.ConversionError` (not a re-declaration).

**Regression**: `ruff`/`mypy` clean; the full real test suite passes
unchanged (no test depended on this import failing).

---

## 3. UTF-8 mojibake in `tin.py`

**Finding**: `vertex_array()`'s own docstring contained `â€”`
(a UTF-8 em-dash re-encoded and misread as Latin-1) instead of a
plain em-dash.

**Evidence**: direct string inspection of the docstring.

**Classification**: purely cosmetic, zero functional impact --
fixed as a trivial, zero-risk correction alongside the others rather
than opened as its own separate cycle.

**Fix**: replaced with a plain `--`.

**Behavior test**: confirmed zero remaining occurrences of the
mojibake sequence in the file.

**Regression**: `ruff` clean; `TIN` still imports and behaves
identically (docstrings don't affect runtime behavior).

---

## 4. `Raster.elevation()` — silent wraparound for out-of-bounds coordinates

**Finding**: `Raster.elevation(x, y)` computed `row`/`column` via
`grid.row(y)`/`grid.column(x)` and indexed `self.values` directly,
with no bounds check.

**Evidence**: for a coordinate just outside the raster's own
negative edge (e.g. `column = -1`), NumPy's own negative-index
wraparound silently returned the value from the OPPOSITE edge of the
raster -- a plausible-looking, entirely wrong number, not an error.
Confirmed directly: `raster.contains(-1.0, 1.0)` correctly reported
`False` for the exact coordinate that `raster.elevation(-1.0, 1.0)`
silently mis-answered. Confirmed the symmetric case (`y` below
`min_y`) behaves the same way. Confirmed `RasterSampler.nearest()` --
a separate utility performing functionally the same nearest-cell
lookup -- already validated `.contains()` first and never had this
problem.

**Classification**: a real, confirmed defect. No test anywhere
exercised an out-of-bounds coordinate for this specific method
(the real project's own `test_raster.py::test_raster_elevation`
only checks in-bounds, grid-aligned points).

**Fix**: `elevation()` now checks `self.contains(x, y)` first and
returns `self.nodata` for anything outside -- matching this class's
own existing `NaN`-as-NoData convention, not inventing a new one.

**Behavior test**: confirmed directly, both edges (`x` and `y`), now
return `NaN` instead of the wraparound value; confirmed every
previously-correct in-bounds case (grid corners and center) returns
byte-for-byte the same value as before the fix.

**Regression**: `ruff`/`mypy` clean; full real test suite passes,
including every `test_raster.py` case.

---

## 5. `TerrainInterpolator.interpolate()` — silent fallback to `NEAREST`

**Finding**: the facade's own dispatch compared `self._method` with
`is` (identity) rather than `==` (value), with an implicit
`NEAREST` fallback for anything that didn't match the other 3
methods by identity.

**Evidence**: a real, ALREADY-FIXED historical bug in this exact
class (2 duplicate `InterpolationMethod` enum definitions, also
compared with `is`) addressed only that specific symptom, not this
dispatch's own structural cause. Confirmed directly: since
`InterpolationMethod` is a `StrEnum`,
`"linear" == InterpolationMethod.LINEAR` is `True` while
`"linear" is InterpolationMethod.LINEAR` is `False` -- Python's own
interpreter emits a `SyntaxWarning` for this exact comparison shape.
`TerrainInterpolator(tin, method="linear").interpolate(0.5, 0.5)`
gave `10.0` (silently `NEAREST`'s own answer) instead of `7.5`
(genuine `LINEAR`), with no error at all.

**Classification**: a real, confirmed defect, and a real,
non-hypothetical risk -- writing a plain string instead of the enum
member is an easy, natural mistake specifically because
`InterpolationMethod` is a `StrEnum`.

**Fix**: dispatch now compares with `==`, and explicitly checks for
`NEAREST` as its own branch rather than treating it as the default;
any value that isn't one of the 4 known members now raises
`InterpolationError` explicitly.

**Behavior test**: confirmed directly, `method="linear"` (the exact
failure case) now gives `7.5`; an unrecognized value now raises
`InterpolationError` with a clear message instead of guessing; all 4
real `InterpolationMethod` members, dispatched through the facade,
confirmed to give identical results to before the fix.

**Regression**: `ruff`/`mypy` clean; full real test suite passes,
including every `test_interpolation_facade.py` case (which already
covers the first, historical bug -- confirmed that fix remains
intact and this one doesn't disturb it).

---

## 6. `weights.barycentric_weights()` — inconsistent exception type

**Finding**: raised a plain `ValueError` for a degenerate triangle,
while the identical underlying problem raises `TriangulationError`
(via `TIN.from_points()`) or `TerrainValidationError` (via
`validate_triangle()`) elsewhere in the same package.

**Evidence**: confirmed directly, calling `barycentric_weights()`
with 3 collinear points raised `ValueError`, while `TIN.from_points()`
given the identical points raised `TriangulationError`.

**Classification**: a real, confirmed inconsistency -- not a
functional bug (the check itself was correct and did fire), but a
genuine mismatch with this package's own established
domain-exception convention, with no test anywhere depending on the
specific `ValueError` type.

**Fix**: now raises `TerrainValidationError`, matching
`validate_triangle()`'s own exception type for what both functions
are, at heart, checking for the same thing.

**Behavior test**: confirmed directly, the same degenerate-triangle
call now raises `TerrainValidationError`; confirmed a valid triangle
still returns identical weights to before the fix.

**Regression**: `ruff`/`mypy` clean; full real test suite passes,
including every `test_weights.py` case (none of which exercised this
specific exception path either before or after).

---

## 7. `contours._levels()` — floating-point drift in elevation labels

**Finding**: the level sequence accumulated via repeated
`level += interval` inside its own loop.

**Evidence**: confirmed directly, for `interval=0.1` (no exact
binary representation) over a realistic 1000-level range, the level
meant to be exactly `100.0` came out as `99.9999999999986`. The
drift was always far smaller than this module's own `EPSILON`
(`1e-9`), so it never changed which triangle edges got contoured --
but the elevation value itself is used verbatim as each contour's
own label in real output (legends, exported layer names).

**Classification**: a real, confirmed defect, cosmetic in effect but
genuine in cause -- no test exercised a non-exact-binary interval
over a long range.

**Fix**: each level is now computed directly as
`first + n * interval` for an integer step count `n` (no
accumulation possible), then rounded to 9 decimal places.

**Behavior test**: confirmed directly, the same 1000-level case now
ends at exactly `100.0`; confirmed an exactly-representable interval
case (`0.5`, 901 levels) and a non-zero `base` offset case are both
unchanged; confirmed the historical exact-vertex-elevation contour
case (see [`contours.md`](./contours.md)) still produces its own
correct, closed contour after this fix.

**Regression**: `ruff`/`mypy` clean; full real test suite passes,
including every `test_contours.py` case.

---

## 8. `validate_triangle()` — collinear-but-distinct vertices not detected

**Finding**: checked only for duplicated vertices, missing the
broader case of 3 genuinely distinct but collinear vertices --
equally degenerate (zero area), just not caught the same way.

**Evidence**: confirmed directly, `validate_triangle()` did not
raise for 3 distinct collinear points, even though `Triangle.area`
for that exact input is `0.0`. The one real caller,
`_geometry.oriented_normal()`, still correctly rejected this case on
its own (via a separate zero-cross-product check further down its
own body) -- but with `TerrainError`, not `TerrainValidationError`,
the same exception `validate_triangle()` itself raises for the
duplicated-vertex case. Confirmed directly: 2 different exceptions
for what a caller would reasonably consider the same category of
problem ("this triangle is degenerate"), depending on *which kind*
of degeneracy it happened to be.

**Classification**: a real, confirmed defect -- not a design choice;
nothing in the codebase or its tests treated the 2-exception split
as intentional.

**Fix**: `validate_triangle()` now also checks `triangle.area <
EPSILON` (this package's own standard tolerance, already used
elsewhere) and raises `TerrainValidationError` for that case too.

**Behavior test**: confirmed directly, both the duplicated-vertex
case and the collinear-but-distinct case now raise
`TerrainValidationError`; confirmed via `oriented_normal()` itself
that both paths now agree; confirmed a genuinely valid triangle is
unaffected.

**Regression**: `ruff`/`mypy` clean; full real test suite passes,
including every `test_validation.py` and `test_slope_aspect.py` case.

---

## 9. `NearestInterpolator.interpolate_many()` — `no-any-return`, a type-only gap

**Finding**: `mypy`, run with the real project's own `pyproject.toml`
against the entire `terrain/` package, reported exactly 1 error
anywhere in the whole package: `nearest.py:91`, `no-any-return` on
`return vz[nearest_index]`.

**Evidence**: a focused mini-audit was performed before any change,
per this project's own established discipline (a confirmed finding
is not automatically a correction) -- isolating the exact expression
with `reveal_type()`. The root cause was confirmed to have nothing
to do with `nearest.py`'s own logic: `vz = vertex_array[:, 2]` is
correctly typed (`vertex_array()`'s own return annotation,
`NDArray[np.float64]`, is precise); `np.argmin(distances, axis=1)`
-- with the real, installed `numpy` 2.4.4's own type stubs -- infers
as `Any` specifically because of the `axis=` keyword argument, not
because of anything this package does. That `Any` then propagates
through the subsequent fancy-indexing (`vz[nearest_index]`) to the
function's own `return` statement. Confirmed, by search, that
`interpolate_many()` has exactly one real caller anywhere in
`terrain/` beyond its own tests -- `LinearInterpolator`, which
delegates to `BarycentricInterpolator`, never to
`NearestInterpolator` -- so `NearestInterpolator.interpolate_many()`
itself has zero internal consumers that could be affected by any
change here.

**Classification**: a real, confirmed finding (not a false positive
of caching, configuration, or any of the other 8 fixes), but purely
a static-typing precision gap in the `numpy` stub ecosystem, not a
runtime defect and not a deliberate contract of any kind. Confirmed
directly: adding an explicit local type annotation alone (no logic
change) resolves the `mypy` error, proving the runtime type was
already correct all along.

**Fix**: the return expression is now bound to an explicitly
annotated local variable before returning:

```python
nearest_index = np.argmin(distances, axis=1)

result: NDArray[np.float64] = vz[nearest_index]

return result
```

No mathematical, algorithmic, or public-API change of any kind --
confirmed identical runtime behavior before and after.

**Behavior test**: confirmed directly, all 6 tests touching
`NearestInterpolator`/`interpolate_many()`
(`test_interpolator_inherits_base_interpolator[NearestInterpolator]`,
`test_interpolate_many_matches_scalar_interpolate[NearestInterpolator]`,
`test_interpolate_many_single_point[NearestInterpolator]`,
`test_interpolate_many_returns_float64_array[NearestInterpolator]`,
`test_nearest_interpolate_many_tie_breaks_to_first_vertex`,
`test_every_method_is_reachable_via_either_import_path[nearest]`)
pass unchanged.

**Regression**: `ruff` clean; `mypy`, with the real project's own
`pyproject.toml`, now reports **0 errors across the entire `terrain/`
package** (32 files) -- the package's own only pre-existing `mypy`
finding is now resolved. Full real test suite: 304 of 304 still
pass. The 8 originally-modified files (see below) re-confirmed clean
alongside this 9th one.

---

## Consolidated regression, all 9 fixes together

- `ruff`, with the real project's own `pyproject.toml`
  (auto-discovered from the package root, not passed via `--config`
  -- confirmed this distinction matters for this specific file):
  clean on all 9 modified files
  (`grid.py`, `terrain/__init__.py`, `tin.py`, `raster.py`,
  `interpolation.py`, `weights.py`, `contours.py`, `validation.py`,
  `nearest.py`).
- `mypy`, same real configuration: clean on the same 9 files,
  confirmed both individually and with `--follow-imports=silent`, and
  confirmed clean across the entire `terrain/` package (32 files)
  together -- **0 `mypy` errors anywhere in `terrain/`**, now that
  fix 9 above resolved the package's own one remaining finding.
- **The real project's own complete test suite: 304 of 304 tests
  passed**, across all 22 real test files for this package, run
  directly against the corrected tree -- the most complete
  regression check performed anywhere in this audit, made possible
  once the environment was reconstructed from a full repository
  export rather than a partial sandbox copy.

## What these 9 fixes do not cover

19 pre-existing `ruff` findings remain, confirmed under the real
project configuration, in files this correction batch did not touch.
They were deliberately left unaddressed, to keep this audit's own
scope to the 9 items above -- documented in full, not hidden, in
[`limitations.md`](./limitations.md). `mypy` now has 0 pre-existing
findings anywhere in `terrain/`, following fix 9.

