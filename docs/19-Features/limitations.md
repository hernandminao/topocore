# Limitations

## Confirmed, deliberately unaddressed: an imprecise log message

`FeatureExtractionManager.detect_all()`'s own non-strict path
(`strict=False`) logs, for every caught `DetectionError`:

```text
Skipping detector '%s': required inputs not available.
```

This message was accurate for `DetectionError`'s own original,
narrower purpose (a detector's `required_inputs` genuinely missing
from the `DetectionContext`). After this audit's own 3 fixes (see
[`validation.md`](./validation.md)), `DetectionError` is now also
raised for non-finite coordinates and other numeric preconditions
that have nothing to do with a missing *input* -- the input (a
`PointCloud`) IS present, its *data* is simply invalid. A detector
skipped for one of these 3 new reasons, under `strict=False`, would
log a message that says its required inputs are unavailable, which
is not what actually happened.

**Confirmed real, deliberately not corrected during this audit**:
this does not affect behavior (the skip-and-continue mechanism itself
works correctly regardless of the message's own wording) and does
not meet the same evidentiary bar the 3 corrected defects did --
those were each confirmed reachable in the real production pipeline
with a concrete, reproducible negative consequence (a raw exception
type escaping this package's own boundary, or a discarded resulting
`Feature`'s own spatial extent). A slightly-imprecise log message,
by contrast, only affects operational diagnosis under a
non-default (`strict=False`) mode -- a real but far smaller cost,
and correcting the message's own wording without a concrete case of
it having caused genuine confusion in practice would risk fixing a
hypothetical rather than a demonstrated problem. Recorded here for a
future, separate decision -- e.g., distinguishing "missing input"
from "invalid input data" in the log message, or including the
exception's own message text in the log line.

## Confirmed pre-existing, out of this audit's own 3-defect scope

`ruff`, with the real project's own `pyproject.toml`, reports 14
findings across `topocore/features/` in files this audit's own 3
fixes never touched (all changes were confined to `_shared.py`):

- **12 `RUF022`/`RUF023`** (`__all__`/`__slots__` not alphabetically
  sorted), across `features/__init__.py` and 7 subpackage
  `__init__.py`/class files. Confirmed, by reading each one, to
  follow the same deliberate semantic/logical grouping already
  established and preserved (with a scoped, justified `# noqa`) in
  `analysis`, `terrain`, and this audit's own edit to `_shared.py`'s
  `__all__` -- not flattened.
- **1 `RUF012`** (`detector.py`'s own `DetectorRegistry._detectors`
  mutable class-level dict): confirmed intentional, directly by that
  class's own docstring -- "A class-level dict, not per-instance
  state: 'what detectors exist' is module-global information" -- a
  genuine global registry, not the accidental shared-mutable-default
  bug this lint rule is designed to catch.

None of these 14 were corrected during this audit, consistent with
its own agreed scope (3 confirmed functional defects, all in
`_shared.py`) -- the same precedent already established during
`16-analysis` (`UP040`) and `16-terrain` (19 `ruff` findings).

## No test suite was available for this package

Confirmed directly, not assumed from an empty search: no
`tests/features/` (or equivalent) directory was present in the
reconstructed repository export this audit worked from, and none was
otherwise provided. Every functional claim in
[`overview.md`](./overview.md) and [`validation.md`](./validation.md)
about this package's own *existing* correctness (the 85/85
`FeatureType`-to-geometry coverage, the 22/22 clean detector
registry, the family-by-family error-path behavior, `models.py`'s
own construction-time validation) rests on this audit's own direct,
hand-verified execution against real code -- not on a pre-existing
test suite this audit could point to. The 11 regression tests this
audit itself added (see `validation.md`) are, as far as this audit
could confirm, the first tests this package has had.

## `core.types`-style open question: none identified here

Unlike `17-core`'s own 12 unconsumed type aliases, this audit did not
identify a comparable "declared but unconnected, possible external
API" question within `topocore.features` itself. The closest
candidates -- `catalogs`/`grammar`/`side` each having their own
`__init__.py` not re-exported from the package root -- were each
confirmed to have real internal consumers
(`feature_builder.py` for `grammar`; `workflow.py` for `side`;
`catalog_audit.py`'s own real, passing self-audit for `catalogs`) and
a coherent, self-documented intent, not left as an open question.

## What this audit did not attempt

A full, detector-by-detector execution of all 22 registered
detectors' own complete parameter space was not performed -- this
audit instead verified the shared infrastructure they all build on
(`models.py`, `_shared.py`, the registry, the 3-family error-path
matrix) exhaustively, and executed representative real detectors
from each family directly, rather than treating 22 detectors as 22
fully independent audits. This is documented as a deliberate
methodology choice in [`overview.md`](./overview.md), not an
oversight -- but it does mean a defect specific to one individual
detector's own particular logic, not shared with its own family,
could exist undiscovered. No evidence of one was found during this
audit's own representative sampling.
