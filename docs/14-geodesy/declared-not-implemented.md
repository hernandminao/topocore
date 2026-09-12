# Declared but Not Implemented

Two types exist in `topocore.geodesy`, are exported from the
package's own `__all__`, and are fully importable and constructible
— but each one's own module docstring explicitly states that its
apparent purpose is not yet backed by real behavior. `VerticalDatum`
used to be listed here too; it has a real consumer now
(`VerticalTransformer` — see
[`vertical-reference.md`](./vertical-reference.md)) and is documented
below only for continuity/context, not as an unimplemented type. This
page exists so that finding one of the genuinely-unimplemented types
in the API doesn't lead to assuming a capability TopoCore doesn't
actually have.

## `LocalCRS`

```python
from topocore.geodesy import LocalCRS
```

Represents a project-local (engineering/construction site)
coordinate system: an origin, rotation, and scale relative to a real
`CRS`.

**Its own module docstring states directly**: *"transforming
coordinates through a `LocalCRS` belongs to a later PR, once
`CoordinateOperation` exists."*

Confirmed directly: `LocalCRS` has zero methods with "transform" in
their name, or any other way to actually apply its own `rotation`/
`scale`/`false_easting`/`false_northing` fields to a coordinate. It
is pure data — you can construct one, read its fields back, and
nothing more.

```python
>>> local = LocalCRS(name="Proyecto Norte", base_crs=CRS.from_epsg(4326), origin_x=0, origin_y=0)
>>> [m for m in dir(local) if "transform" in m.lower()]
[]
```

The class's own docstring is candid about this being intentional
design-freezing, not an oversight: the `rotation` convention
(counter-clockwise from positive X) is "documented now purely to
freeze intent; it has no behavior to validate against yet (no
transform exists)."

## `VerticalDatum`

```python
from topocore.geodesy import VerticalDatum
```

Represents a vertical datum (e.g. EGM96, EGM2008, NAVD88) — pure
data, distinct from `Datum` (horizontal/geometric).

**Update: this now has a real consumer.**
`topocore.geodesy.vertical.VerticalTransformer` accepts a
`source_datum`/`target_datum` pair — see
[`vertical-reference.md`](./vertical-reference.md) for the full,
implemented capability. `VerticalDatum` itself is still pure data
(no method of its own performs a computation); `VerticalTransformer`
carries it purely for traceability and does not branch its own logic
on the specific datum named — confirmed directly, the same
transformation code runs identically regardless of which datum
(EGM96, EGM2008, or a national model such as Colombia's own
GEOCOL2004) is named. The genuine height computation comes from
whichever `GeoidGrid` you provide alongside it, not from
`VerticalDatum` itself.

## `TransformationAccuracy`

```python
from topocore.geodesy import TransformationAccuracy
```

Represents the precision of an actually-executed (or described)
coordinate transformation — explicitly distinguished in its own
module docstring from `ProjectionInfo.accuracy` (the static accuracy
a CRS's own registered operation carries from the EPSG registry).

**Confirmed directly: zero construction sites anywhere in the
codebase outside its own module.** No transformation function in
`topocore.geodesy` — not `CoordinateTransformer`, not any of the
three `transform_*` functions — ever constructs a
`TransformationAccuracy`. It has real validation (`value` must be
non-negative) and could be constructed by calling code, but nothing
in TopoCore itself currently does.

```python
>>> TransformationAccuracy(value=-1.0)
ValidationError: TransformationAccuracy.value cannot be negative, got -1.0.
```

Its own module docstring is explicit about the intended division of
responsibility: *"computing or estimating an accuracy belongs to
whatever produces a `CoordinateOperation`, not to this class"* — that
producer does not yet exist.

## Why this matters

If a future project depends on project-local coordinate systems with
real transformation, vertical datum shifts (e.g. converting GNSS
ellipsoidal heights to a local orthometric benchmark), or reporting
transformation accuracy, none of these are ready-to-use capabilities
today, despite being fully-formed, importable, well-documented
classes. Each represents a genuine, deliberate placeholder for future
work — not a bug, and not something this documentation pass
implements or fixes.
