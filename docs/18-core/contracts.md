# `topocore.core` — Contracts

## The one real cross-project contract `core` provides: `TopoCoreError`

Every domain-specific exception hierarchy in this entire project is
required, by convention, to root itself in `TopoCoreError`. This was
verified directly, not assumed from naming similarity -- by
constructing each domain module's own real exception class and
checking `__bases__ == (TopoCoreError,)` (direct inheritance, not
merely "somewhere in the MRO"):

| Package | Domain base exception | Direct base confirmed |
|---|---|---|
| `alignment` | `AlignmentError` | `TopoCoreError` |
| `gpkg` | `GPKGError` | `TopoCoreError` |
| `processing` | `ProcessingError` | `TopoCoreError` |
| `features` | `FeatureError` | `TopoCoreError` |
| `geodesy` | `GeodesyError` | `TopoCoreError` |
| `survey` | `SurveyError` | `TopoCoreError` |
| `terrain` | `TerrainError` | `TopoCoreError` |
| `workflow` | `WorkflowError` | `TopoCoreError` |
| `io` | `PointCloudIOError` | `TopoCoreError` |
| `io.landxml` | `LandXMLError` | `TopoCoreError` |
| `io.ascii` | `ASCIIError` | `TopoCoreError` |
| `dxf` | `DXFError` | `TopoCoreError` |

**12 of 12, confirmed with zero inconsistency.** This means catching
`topocore.core.exceptions.TopoCoreError` anywhere in consuming code
genuinely catches any domain-specific error raised anywhere in this
project -- a real, working, project-wide contract, not an aspiration
stated only in a docstring. This is also, independently, how
`16-analysis`'s and `16-terrain`'s own audits separately confirmed
`AnalysisError`/`TerrainError` each inherit from their own immediate
parent (`ProcessingError` for `AnalysisError`; `TopoCoreError`
directly for `TerrainError`) -- this document consolidates that
finding at the project-wide level for the first time, rather than
per-package.

## What `core` does NOT provide a cross-project contract for

**`MathError`/`TopologyError`** are not a similar project-wide
convention -- they are specific to `topocore.math`/`topocore.linalg`'s
own domain (numeric/geometric validation), and `TopologyError` in
particular has no real domain attached to it at all (see
[`exceptions.md`](./exceptions.md)).

**`Serializable`** was evidently intended as a structural contract
("anything with `to_wkt()` counts"), but confirmed to have zero real
enforcement anywhere -- no function signature, return type, or
`isinstance` check anywhere in the project requires or checks for it,
in either its `core` or `geometry` form. See
[`protocols.md`](./protocols.md).

**`core.types`'s own aliases** are not a project-wide contract in the
same sense `TopoCoreError` is -- they are convenience type aliases
adopted selectively (5 of 17, confirmed) by specific consuming
packages, not a requirement anything in the project enforces. See
[`types.md`](./types.md).

## Consumption pattern, confirmed consistent across the whole project

Every real consumer of anything in `core` -- exceptions, types alike
-- imports directly from the specific submodule
(`topocore.core.exceptions`, `topocore.core.types`), never from
`topocore.core` itself. This matches `core/__init__.py`'s own
minimal export surface (`__version__` only) exactly, and matches the
project's own top-level `topocore/__init__.py` doing the identical
thing. See [`overview.md`](./overview.md) for why this is confirmed
deliberate.
