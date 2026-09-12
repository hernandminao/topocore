# Validation — 0 corrections, and why

## Methodology, identical to `16-analysis`/`16-terrain`

```text
finding -> evidence -> classification -> fix (only if justified) -> behavior test -> regression
```

`core` went through the identical 5-point audit structure as
`terrain` (inventory, contracts/consumers, functional integration,
error paths, regression), scaled to `core`'s own much smaller real
surface (6 files, 90 lines, no algorithms). The outcome is genuinely
different from `16-analysis` (8 defects) and `16-terrain` (9
defects): **0 functional defects were found**, and consequently **0
corrections were applied**.

## Why 0, not "audit incomplete"

This was not a shorter or less rigorous audit -- every one of
`core`'s own 6 files was read in full, and every real consumer
relationship was checked by search across the project's complete
real source (406 files):

- `TopoCoreError`'s own inheritance contract was checked against all
  12 real domain exception hierarchies individually, by direct
  construction and `__bases__` inspection, not by name-matching --
  confirmed 12 of 12 correct.
- `MathError`'s own real usage was confirmed with a live,
  reproducible case (`Point3D` rejecting `NaN`).
- Every one of `core.types`'s 17 aliases was checked for real
  importers by direct search, not sampled.
- `Serializable` (both copies) was checked for real usage via 3
  independent search patterns (direct import, type-hint annotation,
  `isinstance` check) -- all 3 came back empty for both copies.
- `TopologyError` was checked for `raise`/`except`/import sites --
  all empty -- and confirmed, by the project's own author directly,
  to have no test anywhere either.

**A defect requires a confirmed, real, functioning contract that
code violates.** `core` has exactly one such real, working contract
(`TopoCoreError`'s own inheritance chain), and it holds with zero
exceptions across all 12 real consumers. Everything else this audit
found was either working correctly (`MathError`, `__version__`, the
5 real `core.types` aliases) or genuinely unconnected to anything
(the 3 dead-code items, the 2 refactor opportunities, the 12 open
`core.types` aliases) -- unconnected code is not, by itself, a
defect to fix; it is a documentation and future-decision matter,
covered in full in [`limitations.md`](./limitations.md).

## What was deliberately NOT done, and why

**`DEFAULT_ENCODING`/`EPSG_WGS84` centralization was not applied.**
Confirmed real, recurring hardcoded literals exist in 12 and 3 other
files respectively -- but nothing in the project's own source
establishes `core.constants` as the mandatory source for these
values, and applying this centralization would mean editing 15 files
across packages entirely outside `core`'s own scope, on this audit's
own initiative, without those packages' own audits having reached
that point yet. Recorded as an opportunity in
[`limitations.md`](./limitations.md), not applied unilaterally.

**The 3 confirmed dead-code items were not deleted.** Per the
standing project-wide precedent (established explicitly after the
`16-analysis` `ComparisonError` episode, and followed again in
`16-terrain`): confirmed absence of a consumer, even confirmed
directly by the project's own author with no remaining ambiguity, is
documented as a removal candidate, not acted on unilaterally during a
documentation audit. Removal is a deliberate decision for the
project's own author to make explicitly, separate from PR22's own
documentation scope.

**The `core.types` 12-alias question was not resolved either way.**
Confirmed no internal consumer; confirmed NOT the same as
`TopologyError`/`Serializable`, because external usage outside this
repository could not be ruled out by a source-only audit. Recorded
as an open question in [`overview.md`](./overview.md) and
[`limitations.md`](./limitations.md), not classified as dead code.

## Regression

No code was modified, so "regression" here confirms the audit
process itself introduced no side effects -- relevant specifically
because this audit's own environment required installing several
real project dependencies (`shapely`, `pyproj`, `laspy`, `pye57`)
partway through, to complete the `TopoCoreError` inheritance check
across all 12 packages. The full, real `16-terrain` test suite (304
tests) was re-run afterward and confirmed unaffected: 304 of 304
still pass.
