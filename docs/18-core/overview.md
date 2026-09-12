# `topocore.core` — Overview

## What this document is built from

`topocore.core` is small (6 files, 90 lines total) and was audited
completely, not sampled: every line of every file was read, every
claim below was checked by direct execution or by search across the
project's own real, complete source (406 files, reconstructed from a
full repository export). No test file for `core` was available in
that export -- confirmed directly by the project's own author, not
assumed from its absence -- so every "zero consumers" claim below
means exactly that: zero real callers anywhere in the source, not
"zero visible to this audit."

## What actually belongs to `core`

```text
topocore/core/
├── __init__.py       exports only __version__
├── version.py        __version__ = "0.1.0"
├── exceptions.py      TopoCoreError, MathError, TopologyError
├── constants.py        DEFAULT_ENCODING, PROJECT_NAME, EPSG_WGS84
├── protocols.py          Serializable
└── types.py               17 type aliases (scalars + NumPy arrays)
```

Nothing else. `core` has no classes with behavior, no algorithms, no
validation functions -- it is purely foundational declarations that
other packages build on.

## Why `core/__init__.py` exports only `__version__`

**Confirmed deliberate, not an oversight**: `topocore/__init__.py`
itself -- the project's own top-level package, one level above
`core` -- follows the identical pattern, exporting only
`__version__` and nothing else. This is the project's own consistent
convention throughout: every consumer imports directly from the
specific submodule it needs
(`from topocore.core.exceptions import TopoCoreError`, confirmed the
real pattern used by all 12 real exception hierarchies -- see
[`exceptions.md`](./exceptions.md)), not from a flattened package
root. `core`'s own minimalism is this same convention applied to
itself, not a gap.

## What has real, confirmed consumers

- **`TopoCoreError`**: the base exception for the entire project.
  Confirmed directly -- all 12 domain-specific exception hierarchies
  in the codebase (`alignment`, `gpkg`, `processing`, `features`,
  `geodesy`, `survey`, `terrain`, `workflow`, `io`, `io.landxml`,
  `io.ascii`, `dxf`) inherit from it directly, with no exceptions and
  no inconsistency in the inheritance chain. See
  [`exceptions.md`](./exceptions.md) and [`contracts.md`](./contracts.md).
- **`MathError`**: used consistently across `topocore.math` and
  `topocore.linalg`, confirmed with a real, reproducible case
  (`Point3D` rejecting a `NaN` coordinate raises it directly).
- **5 of `core.types`'s own 17 type aliases** (`FloatArray1D`,
  `FloatArray3D`, `IntArray1D`, `PointAttributeArray`, `Vector3D`):
  confirmed real imports in `processing`, `features`, `geodesy`,
  `pointcloud`. See [`types.md`](./types.md).
- **`__version__`**: confirmed to propagate correctly to
  `topocore.__version__`, and consumed by `analysis`'s own
  `__init__.py` in 2 places.

## What was confirmed as dead code -- not removed during this audit

Investigated with the same discipline established during the
`16-analysis`/`16-terrain` audits (a confirmed absence of consumers
is a candidate, not an automatic deletion): each of these 3 was
checked for real callers, real `raise`/`except` sites, real
`isinstance`/type-hint usage, and (per the project's own author) has
no test anywhere either.

- **`TopologyError`**: declared, never raised, never caught, never
  referenced anywhere beyond its own definition.
- **`core.protocols.Serializable`**: declared, never imported, never
  type-hinted against, never checked with `isinstance`. A
  structurally identical `Serializable` exists in
  `geometry.protocols` (confirmed a deliberate, correct placement,
  not a duplication to resolve -- it lives alongside `HasArea`/
  `HasLength`/`HasVolume`/`HasCentroid`/`Bounded`, a coherent,
  documented set of geometry-specific protocols). `core`'s own copy
  is the abandoned original, not a second intentional design. See
  [`protocols.md`](./protocols.md).
- **`PROJECT_NAME`** (`constants.py`): the string `"TopoCore"` does
  not appear anywhere else in the codebase, not even as a hardcoded
  literal -- unlike the 2 items below, there is no real, recurring
  need this constant addresses.

None of these 3 were removed during PR22. They are documented here,
in full, for a future, deliberate cleanup decision -- not silently
carried forward and not deleted without that explicit decision. See
[`limitations.md`](./limitations.md).

## What was confirmed as a real gap, but not a defect: 2 refactor opportunities

`DEFAULT_ENCODING` (`"utf-8"`) and `EPSG_WGS84` (`4326`) are each
confirmed to address a genuinely recurring real need -- their own
literal values are hardcoded independently in 12 and 3 other files
respectively, rather than importing the shared constant. This is
**not classified as a functional defect**: a hardcoded `"utf-8"` or
`4326` is not wrong, and nothing in the project's own source or
structure establishes that `core.constants` was ever meant to be the
single mandatory source for these values. Recorded as a
centralization opportunity, not corrected during this audit -- doing
so would mean touching 15 files across packages this audit did not
otherwise scope, well beyond what a `core`-focused audit should
decide unilaterally. See [`limitations.md`](./limitations.md).

## `core.types`'s own 12 unconsumed aliases — an open question, not a finding either way

12 of the 17 type aliases in `core.types` (`Coordinate`, `Elevation`,
`Distance`, `Angle`, `FloatArray2D`, `IntArray2D`, `UInt8Array1D`,
`UInt16Array1D`, `UInt32Array1D`, `BoolArray1D`, `Matrix3x3`,
`Matrix4x4`) have no confirmed internal consumer anywhere in this
project's own source. **This is stated deliberately as an open
question, not as dead code**: 12 of 17 aliases lack an internal
consumer identified in the repository audited; whether they are
consumed by external code depending on this package could not be
verified from the repository alone. Confirmed, separately, that at
least one of these exact names (`Coordinate`, `Elevation`,
`Distance`) is independently re-declared, identically, in
`terrain.types` rather than imported from here -- evidence of a
missed centralization opportunity for that specific package, not
evidence that the alias itself is unused everywhere. These 12
aliases should NOT be removed on the strength of this audit's own
internal-consumer search alone. See [`types.md`](./types.md) and
[`limitations.md`](./limitations.md).

## What this audit did not do

Did not modify any code -- 0 corrections were applied to `core`,
because 0 functional defects were found. Did not attempt to resolve
the `core.types` external-usage question definitively -- that
requires information (public release history, external dependents)
outside what a source-code audit alone can establish. Did not treat
`QA/Validation` or `Optimization` as `core`-equivalent documentation
targets -- both are cross-cutting PR22 processes/methodology, to be
documented as such when that block of work is reached, not as a
`topocore` package.

## Where to go next

- [`types.md`](./types.md) -- the full 17-alias inventory, 5 real / 12 open.
- [`exceptions.md`](./exceptions.md) -- `TopoCoreError`/`MathError`/`TopologyError`.
- [`protocols.md`](./protocols.md) -- `Serializable`, both copies.
- [`contracts.md`](./contracts.md) -- the cross-project exception-inheritance
  contract, confirmed for all 12 real hierarchies.
- [`validation.md`](./validation.md) -- this audit's own methodology and
  its explicit "0 corrections" outcome, with the same evidence
  discipline as `16-analysis`/`16-terrain`.
- [`limitations.md`](./limitations.md) -- the complete, itemized account
  of confirmed dead code, refactor opportunities, and the open
  `core.types` question.
