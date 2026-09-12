# Primitives — geometry and manual data construction

Found missing during a revalidation pass over the rest of this
guide: every other document uses `Point3D`, `PointCloud`, and
`Chunk` constantly (`TIN.from_points()`, `Feature.geometry.vertices`,
every reader's own `.read()` return value) without ever showing how
to construct one yourself. This page fills that gap, verified by
real execution end to end.

## `Point3D` / `Point2D`

```python
from topocore.geometry.point3d import Point3D

p = Point3D(x: float, y: float, z: float)
p.x, p.y, p.z
```

Frozen (`@dataclass(frozen=True, slots=True)`). Confirmed directly
(see [`terrain.md`](./terrain.md)'s own audit history): rejects
non-finite coordinates unconditionally, raising `MathError` -- there
is no way to construct a `Point3D` with a `NaN`/`Inf` component.

## `BBox3D` / `BBox2D` — axis-aligned bounding boxes

```python
from topocore.geometry.bbox3d import BBox3D

box = BBox3D(min_x: float, min_y: float, min_z: float, max_x: float, max_y: float, max_z: float)
```

## `Vector3D` / `Vector2D`

```python
from topocore.linalg.vector3d import Vector3D

v = Vector3D(x: float, y: float, z: float)
```

## Building a `PointCloud` from raw arrays — no file required

Every reader in [`point-clouds.md`](./point-clouds.md) ultimately
produces a `PointCloud` this same way. Confirmed by real execution:

```python
import numpy as np
from topocore.pointcloud.pointcloud import PointCloud
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.attributes import PointAttribute

n = 5
chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
chunk[PointAttribute.X][:] = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
chunk[PointAttribute.Y][:] = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
chunk[PointAttribute.Z][:] = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

cloud = PointCloud()
cloud.add_chunk(chunk)

cloud.point_count   # 5
```

`Chunk(size, attributes, *, source_id: int = 0)` requires `X`, `Y`,
`Z` among `attributes` -- confirmed directly, constructing a `Chunk`
missing any of the 3 raises `ValueError` naming which are missing.
Additional attributes from the full list in
[`point-clouds.md`](./point-clouds.md) (`INTENSITY`, `COLOR`, ...)
are added the same way -- include the `PointAttribute` in the list
passed to `Chunk()`, then assign into `chunk[PointAttribute.X]` (a
real NumPy array view, sliceable and assignable in place).

## Building a `TIN` from your own points

Combines directly with [`terrain.md`](./terrain.md)'s own
`TIN.from_points()`:

```python
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

points = (
    Point3D(0.0, 0.0, 10.0),
    Point3D(10.0, 0.0, 12.0),
    Point3D(10.0, 10.0, 15.0),
    Point3D(0.0, 10.0, 11.0),
)
tin = TIN.from_points(points)
```

## Where to go next

- [`point-clouds.md`](./point-clouds.md) -- reading a real file
  instead of constructing one by hand.
- [`terrain.md`](./terrain.md) -- what `TIN.from_points()` does with
  the points once you have them.
- [`features.md`](./features.md) -- `FeatureGeometry.vertices` uses
  the same `(n, 3)` NumPy array shape as a `Chunk`'s own per-attribute
  arrays.
