# Limitations

## Confirmed dead code — 3 items, documented, not removed

Each confirmed by direct search across the project's complete real
source, plus, per the project's own author, confirmed absent from
the real test suite as well (not merely absent from this audit's own
source-only export).

| Item | Where | Evidence |
|---|---|---|
| `TopologyError` | `core/exceptions.py` | Zero `raise`, zero `except`, zero reference anywhere beyond its own declaration; zero tests |
| `Serializable` | `core/protocols.py` | Zero imports, zero type-hints, zero `isinstance` checks anywhere; zero tests. (Its structurally-identical counterpart in `geometry.protocols` is confirmed the correct, intentional version -- see [`protocols.md`](./protocols.md) -- and is explicitly NOT included in this dead-code classification) |
| `PROJECT_NAME` | `core/constants.py` | The string `"TopoCore"` does not appear anywhere else in the codebase, not even as a hardcoded literal -- no recurring need identified at all |

**Not removed during PR22.** Following the project's own established
precedent (a confirmed absence of consumers is a removal candidate,
not an automatic deletion -- see [`validation.md`](./validation.md)
for the full reasoning), these remain in the codebase, fully
documented, awaiting a separate, deliberate decision.

## Refactor opportunities — 2 items, real need, not a defect

| Item | Value | Reinvented as a literal in |
|---|---|---|
| `DEFAULT_ENCODING` | `"utf-8"` | 12 other files |
| `EPSG_WGS84` | `4326` | `gpkg/exporter.py`, `geodesy/operation.py`, `workflow/workflow.py` |

Both address a real, recurring need -- unlike the 3 dead-code items
above, these values genuinely matter and are genuinely used, just
not from this shared location. **Not classified as a functional
defect**: nothing in the project's own source establishes
`core.constants` as the required, single source for these values: a
hardcoded `"utf-8"` or `4326` is not incorrect. Centralizing them
would mean editing up to 15 files across multiple packages, which is
outside this audit's own `core`-focused scope. Recorded as an
opportunity for a future, deliberate refactor, not applied
unilaterally here.

## `core.types` — 12 of 17 aliases with no internal consumer, classified as an open question

```text
Coordinate, Elevation, Distance, Angle,
FloatArray2D, IntArray2D,
UInt8Array1D, UInt16Array1D, UInt32Array1D,
BoolArray1D, Matrix3x3, Matrix4x4
```

**Deliberately NOT classified as dead code**, unlike the 3 items
above. The distinction: `TopologyError`/`core.protocols.Serializable`/
`PROJECT_NAME` were each confirmed, additionally, to have no test
anywhere in the real project (per the project's own author) and no
plausible external purpose (`PROJECT_NAME`'s own value never appears
anywhere at all). The 12 aliases here have neither of those
additional confirmations -- only "no internal consumer identified in
this repository," which does not rule out external consumers of a
published package. The accurate, limited claim: **12 of 17 aliases
in `core.types` have no consumer identified within the audited
repository; whether external code depends on them could not be
verified from the repository alone.**

One partial, related, separate finding: `terrain.types` independently
declares `Elevation`/`Distance`/`Coordinate` with identical
definitions (`= float`), rather than importing them from `core.types`.
This confirms a missed centralization opportunity specific to
`terrain`'s own package -- it does not, by itself, establish that
`core.types`'s own versions are unused everywhere, since `terrain`'s
choice to duplicate rather than import says nothing about whether
some other, external consumer imports the `core.types` version
directly. See [`types.md`](./types.md) for the complete inventory,
including the 8 array-type aliases that, despite having no
independent consumer of their own, ARE collectively referenced by
`PointAttributeArray`'s own union type -- itself a real, confirmed
consumer (`pointcloud.chunk`).

## What this audit explicitly recommends, without deciding unilaterally

1. **A deliberate decision on the 3 confirmed dead-code items**
   (remove, or keep with documentation) -- this audit supplies the
   complete evidence; the decision itself belongs to the project's
   own maintainer.
2. **A deliberate decision on whether to centralize
   `DEFAULT_ENCODING`/`EPSG_WGS84`** -- if yes, this would be its own
   correction batch spanning up to 15 files outside `core`, not a
   `core`-only change.
3. **Clarifying `core.types`'s own intended audience** -- if this
   package is distributed externally and the 12 aliases are part of
   its own advertised public API, they should be documented as such
   explicitly (and perhaps given their own `__all__`); if not, they
   become dead-code candidates on the same footing as the 3 items
   above, pending that same explicit decision.

None of these 3 recommendations were acted on during this specific
audit -- consistent with `core`'s own final tally of 0 corrections
applied, documented in full in [`validation.md`](./validation.md).
