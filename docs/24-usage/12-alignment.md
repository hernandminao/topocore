# Alignment — horizontal and vertical

`topocore.alignment` is confirmed, per its own module docstring, a
**first-class geometric domain in its own right, independent of the
LandXML format** -- not merely an internal detail of
[`landxml.md`](./landxml.md)'s own `<Alignments>` support, even
though that's the interoperability need it was originally built for.

## Horizontal elements

3 element types, all chained into an `Alignment`:

```python
from topocore.geometry.point2d import Point2D
from topocore.alignment.elements import LineElement, ArcElement, SpiralElement
```

```python
LineElement(start: Point2D, end: Point2D)
```
A straight segment. Confirmed rejected at construction if `start`
and `end` coincide (a zero-length line).

```python
ArcElement(start: Point2D, end: Point2D, center: Point2D, radius: float, clockwise: bool, *, tolerance: float | None = None)
```
A circular arc. `start`/`end` must both sit at `radius` from
`center` -- confirmed validated at construction, not assumed. A
full-circle arc (`start == end`) is confirmed unsupported (ambiguous
sweep direction/length from geometry alone) -- a deliberate, documented
limitation.

**`clockwise` determines a full 0-360° sweep, including "major" arcs
past 180° -- confirmed by real execution, not obvious from the
parameter name alone.** The same `start`/`center`/`end` pair, with
only `clockwise` flipped, gave 2 completely different real results:

```python
# center=(0,10), start=(0,0), end=(10,10), radius=10 in both cases
ArcElement(..., clockwise=True)    # confirmed: sweeps the MAJOR arc, 270°
ArcElement(..., clockwise=False)   # confirmed: sweeps the MINOR arc, 90°
```

If your intended arc doesn't match what you get, this is almost
certainly why -- check the actual sweep, don't assume `clockwise`
alone picks "the short way around."

`tolerance` (construction-time only) exists specifically for
`LandXMLReader`'s own use: confirmed, real-world LandXML exports from
Autodesk Civil 3D have `start`/`center`/`end`/`radius` values that
are only mutually consistent to ~1e-8, not this domain's own default
~1e-9 -- a property of serialized engineering data, not a TopoCore
precision guarantee. Leave it `None` for direct API use.

```python
SpiralElement(start: Point2D, end: Point2D, pi: Point2D, radius_start: float, radius_end: float, length: float, clockwise: bool, *, tolerance: float | None = None)
```
A clothoid (Euler spiral) transition. Confirmed required: exactly
one of `radius_start`/`radius_end` must be `math.inf` (the
"straight" end) -- both infinite is rejected (that's a `LineElement`,
not a spiral), and both finite (a curve-to-curve compound spiral) is
confirmed explicitly out of scope, "a real but comparatively rare
case... deserves its own verification pass," not silently folded in.
`pi` (the tangent intersection point) is stored but not
cross-validated against `radius`/`length` -- informational only, per
its own docstring.

## `Alignment` — chaining elements, confirmed with real execution

```python
from topocore.alignment.models import Alignment

Alignment(name: str, elements: tuple[HorizontalElement, ...], start_station: float = 0.0, desc: str | None = None)

alignment.length         # property
alignment.end_station     # property
alignment.station_to_point(station: float) -> Point2D
```

`elements` must chain continuously -- element `i`'s own `end` must
equal element `i+1`'s own `start` -- confirmed enforced at
construction, not silently accepted if discontinuous.

Confirmed with a real, corrected example (a 90° arc followed by a
straight segment):

```python
arc = ArcElement(start=Point2D(0.0, 0.0), end=Point2D(10.0, 10.0),
                  center=Point2D(0.0, 10.0), radius=10.0, clockwise=False)
line = LineElement(start=Point2D(10.0, 10.0), end=Point2D(60.0, 10.0))

alignment = Alignment(name="EJE_VIA_1", elements=(arc, line))
alignment.length                    # confirmed real output: 65.71 (15.71 arc + 50.0 line)
alignment.station_to_point(20.0)    # confirmed real output: Point2D(x=14.29, y=10.0) --
                                     # 4.29m into the straight segment past the arc's end
```

## Vertical elements and `DesignProfile`

```python
from topocore.alignment.vertical_elements import GradeSegment, VerticalCurve
from topocore.alignment.models import DesignProfile
```

```python
GradeSegment(start_station: float, end_station: float, start_elevation: float, end_elevation: float)
```
A constant-slope tangent segment.

```python
VerticalCurve(pvi_station: float, pvi_elevation: float, incoming_grade: float, outgoing_grade: float, length_in: float, length_out: float)
```
A parabolic vertical curve (LandXML `<ParaCurve>`), symmetric or
asymmetric (`length_in == length_out` or not) -- confirmed
implementing the Hickerson (1964) method per its own docstring,
verified there against 2 worked examples.

```python
DesignProfile(alignment_name: str, elements: tuple[VerticalElement, ...], desc: str | None = None)

profile.start_station   # property
profile.end_station      # property
profile.elevation_at(station: float) -> float
```

`alignment_name` links a `DesignProfile` to its `Alignment` **by
name, not by object reference** -- confirmed deliberate, avoiding a
reference cycle between the 2 classes. `elements` must chain
continuously in station, elevation, **and** grade (slope) --
confirmed: a slope discontinuity between a grade segment and an
adjoining vertical curve is rejected at construction, not physically
valid.

Confirmed with real execution:

```python
segment = GradeSegment(start_station=0.0, end_station=100.0, start_elevation=10.0, end_elevation=15.0)
profile = DesignProfile(alignment_name="EJE_VIA_1", elements=(segment,))

profile.elevation_at(50.0)   # confirmed real output: 12.5 -- exact midpoint of a linear grade
```

## Where this connects

- [`landxml.md`](./landxml.md) -- `NamedAlignment` wraps an
  `Alignment` (plus an optional `DesignProfile`) for LandXML
  `<Alignments>` round-tripping -- confirmed with a real 2-segment
  `LineElement` chain written and read back intact.
- Only `ArcElement`/`SpiralElement` matching `<Curve crvType="arc">`/
  `<Spiral spiType="clothoid">` are read from a real LandXML file per
  `topocore.io.landxml`'s own documented scope -- everything else in
  this page (`Alignment`, `DesignProfile`, direct construction) is
  format-independent and usable without LandXML at all.
