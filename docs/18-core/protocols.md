# `core.protocols` — `Serializable`, and its real counterpart in `geometry`

## `core.protocols.Serializable` — confirmed dead code

```python
class Serializable(Protocol):
    def to_wkt(self) -> str: ...
```

The entire file. No module docstring, no class docstring, no
sibling protocols -- confirmed the only declaration in this file.
Confirmed, by direct search across the whole project: zero imports
of it, zero type-hints against it, zero `isinstance()` checks against
it, anywhere. Confirmed, by the project's own author: no test
anywhere in the real repository exercises it either.

## `geometry.protocols.Serializable` — confirmed the real, deliberate version

```python
class HasArea(Protocol):
    @property
    def area(self) -> float: ...

class HasLength(Protocol):
    @property
    def length(self) -> float: ...

class HasVolume(Protocol):
    @property
    def volume(self) -> float: ...

class HasCentroid(Protocol):
    @property
    def centroid(self) -> Point2D | Point3D: ...

class Bounded(Protocol):
    def bounding_box(self) -> BBox2D | BBox3D: ...

class Serializable(Protocol):
    def to_wkt(self) -> str: ...
```

Structurally identical to `core`'s own copy (`to_wkt(self) -> str`)
-- but confirmed a deliberate, correct placement, not a duplication
to resolve: this file carries its own module docstring ("Structural
typing protocols shared across geometry primitives"), and
`Serializable` sits alongside 5 other geometry-specific protocols
(`HasArea`, `HasLength`, `HasVolume`, `HasCentroid`, `Bounded`) as
one coherent, purpose-built set -- confirmed by the project's own
author to be the correct version, not itself a finding requiring
further investigation.

**Confirmed, like its `core` counterpart, to have zero real importers
or type-hint usage anywhere in the project** -- this is true of both
copies equally, and is not, by itself, evidence against
`geometry.protocols.Serializable`'s own legitimacy (it belongs to a
real, coherent, evidently intentional set; `core`'s copy belongs to
nothing).

## The real concept behind `Serializable` does exist and is used -- just not through this protocol

Confirmed by direct search: exactly one real `to_wkt()` implementation
exists anywhere in the codebase, on `geodesy.crs.CRS`. `CRS` therefore
structurally satisfies both `Serializable` definitions (they are
identical), without either one ever being referenced as a type
constraint anywhere. The underlying idea -- a WKT-serializable
geospatial object -- is real and has at least one real
implementation; the `Protocol` built to describe it was simply never
adopted anywhere as an actual contract.

## Classification

`core.protocols.Serializable` is classified the same way as
`TopologyError` (see [`exceptions.md`](./exceptions.md)): confirmed
dead code, not removed during this audit, recorded for a future,
deliberate cleanup decision. `geometry.protocols.Serializable` is
explicitly NOT part of that classification -- it is the correct,
intentional version, confirmed as such, and this audit's own
findings about zero usage apply equally to it only in the narrow,
factual sense (no current consumer), not as a judgment on its own
legitimacy. See [`limitations.md`](./limitations.md).
