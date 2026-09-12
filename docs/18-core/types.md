# `core.types` — 17 type aliases

All declared with PEP 695 `type` statements (the project's minimum
Python is `3.12`, confirmed in the real `pyproject.toml`). No
`__all__` in this file -- confirmed, and consistent with the rest of
`core`'s own submodules (`constants.py`, `exceptions.py`,
`protocols.py` have none either); every consumer imports specific
names directly (`from topocore.core.types import FloatArray1D`), not
a wildcard.

## Scalar types

```python
type Coordinate = float
type Elevation = float
type Distance = float
type Angle = float
```

**Confirmed zero internal consumers, for all 4**, by direct search
across the project's own complete source. **Confirmed, separately**:
`terrain.types` independently declares `type Elevation = float`,
`type Distance = float`, `type Coordinate = float` -- identical
definitions, not imported from here. This is direct evidence that at
least `terrain` reinvented rather than centralized; it is not
evidence that these 4 aliases in `core` are themselves unused by
anything outside this repository. See
[`overview.md`](./overview.md)'s own framing of this as an open
question.

## NumPy array types

```python
type FloatArray1D = NDArray[np.float64]
type FloatArray2D = NDArray[np.float64]
type FloatArray3D = NDArray[np.float64]

type IntArray1D = NDArray[np.int64]
type IntArray2D = NDArray[np.int64]

type UInt8Array1D = NDArray[np.uint8]
type UInt16Array1D = NDArray[np.uint16]
type UInt32Array1D = NDArray[np.uint32]

type BoolArray1D = NDArray[np.bool_]

type Matrix3x3 = NDArray[np.float64]
type Matrix4x4 = NDArray[np.float64]

type Vector3D = NDArray[np.float64]
```

**Confirmed real internal consumers, by direct `from topocore.core.types
import ...` search**:

| Alias | Confirmed importer(s) |
|---|---|
| `FloatArray1D` | `processing._shared`, `features._shared`, `features.models`, `features.protocols`, `geodesy.georeferencing.apply` |
| `FloatArray3D` | `processing._shared` |
| `IntArray1D` | `features._shared`, `features.models` |
| `Vector3D` | `processing.types` |
| `PointAttributeArray` | `pointcloud.chunk` |

**Confirmed zero internal consumers**: `FloatArray2D`, `IntArray2D`,
`UInt8Array1D`, `UInt16Array1D`, `UInt32Array1D`, `BoolArray1D`,
`Matrix3x3`, `Matrix4x4` -- 8 of the 12 array-type aliases.

## `PointAttributeArray` -- a union of every array type above

```python
type PointAttributeArray = (
    FloatArray1D | FloatArray2D | FloatArray3D
    | IntArray1D | IntArray2D
    | UInt8Array1D | UInt16Array1D | UInt32Array1D
    | BoolArray1D
    | Matrix3x3 | Matrix4x4
    | Vector3D
)
```

Confirmed a real consumer: `pointcloud.chunk` imports this directly,
consistent with its own conceptual role -- a point cloud attribute
column can legitimately be any of the numeric/boolean array shapes
this union covers. Its own definition referencing the 8 "unconsumed"
aliases above does not, by itself, make them dead: this union type
IS a real, live consumer of all 12 array aliases collectively, even
for the 8 that have no OTHER, independent consumer of their own.
Worth noting precisely: this changes the array-alias picture from
"8 of 12 unused" to "8 of 12 have no consumer other than this one
union type" -- a real distinction, recorded here rather than
overstated either direction.

## Summary

| Category | Total | Confirmed real consumer | No independent consumer found |
|---|---|---|---|
| Scalar (`Coordinate`, `Elevation`, `Distance`, `Angle`) | 4 | 0 | 4 |
| Array types | 12 | 4 directly + all 12 via `PointAttributeArray`'s own union | 8 (no consumer of their own, beyond that union) |
| `PointAttributeArray` itself | 1 | 1 (`pointcloud.chunk`) | -- |

See [`limitations.md`](./limitations.md) for how this audit
classifies the "no independent consumer" cases -- not as confirmed
dead code, unlike `TopologyError`/`core.protocols.Serializable`/
`PROJECT_NAME` (see [`exceptions.md`](./exceptions.md) and
[`protocols.md`](./protocols.md)), because external usage outside
this repository could not be ruled out.
