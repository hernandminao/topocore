# LandXML

`topocore.io.landxml` reads and writes LandXML 1.2 -- confirmed
covering `<Surfaces>` (TIN), `<CgPoints>` (survey points), and
`<Alignments>` (`<CoordGeom>` + optional `<Profile>`/`<ProfAlign>`).
`<Feature>` embedded in a `<CgPoint>` is confirmed explicitly out of
scope -- TopoCore has no domain model for it, and this module's own
docstring states it does not invent one just to round-trip it.

Unlike every other reader in [`point-clouds.md`](./point-clouds.md),
`LandXMLReader.read()` returns a `LandXMLDocument`, not a
`PointCloud`/`SurveyPointSet` directly -- confirmed by design, since
a single LandXML file can carry surfaces, point groups, and
alignments together. `Workflow` does not integrate LandXML directly
-- confirmed by search, no stage method in
[`workflows.md`](./workflows.md) reads or writes it.

## The document model

```python
from topocore.io.landxml import LandXMLDocument, LinearUnit, NamedSurface, NamedPointGroup, NamedAlignment

LandXMLDocument(
    surfaces: tuple[NamedSurface, ...] = (),
    point_groups: tuple[NamedPointGroup, ...] = (),
    alignments: tuple[NamedAlignment, ...] = (),
    crs: str | None = None,
    linear_unit: LinearUnit = LinearUnit.METER,   # METER | FOOT
)

NamedSurface(name: str, tin: TIN, desc: str | None = None)
NamedPointGroup(name: str, points: SurveyPointSet, desc: str | None = None)
NamedAlignment(name: str, alignment: Alignment, profile: DesignProfile | None = None, desc: str | None = None)
```

`name` must be unique within each of the 3 collections -- confirmed
by real execution: 2 surfaces named `"DUPLICADA"` raise
`LandXMLValidationError` ("Duplicated Surface name(s): ['DUPLICADA']")
at write time, before anything is written to disk.

## Reading and writing — confirmed with a real round trip

```python
from topocore.io.landxml import LandXMLReader, LandXMLWriter

report = LandXMLWriter(path, coordinate_precision=10).write(document)   # -> LandXMLWriteReport
document = LandXMLReader(path).read()                                    # -> LandXMLDocument
```

Confirmed real, complete round trip -- a `NamedSurface` (4-vertex TIN)
and a `NamedPointGroup` (2 survey points with field codes), written
and read back with every value intact:

```python
from topocore.io.landxml import LandXMLDocument, LandXMLWriter, LandXMLReader, NamedSurface, NamedPointGroup, LinearUnit
from topocore.terrain.tin import TIN
from topocore.geometry.point3d import Point3D
from topocore.survey.models import SurveyPointSet, SurveyPoint

tin = TIN.from_points((
    Point3D(0.0, 0.0, 10.0), Point3D(10.0, 0.0, 12.0),
    Point3D(10.0, 10.0, 15.0), Point3D(0.0, 10.0, 11.0),
))
surface = NamedSurface(name="EXISTENTE", tin=tin, desc="Superficie de terreno existente")

points = SurveyPointSet(points=(
    SurveyPoint(id="1", x=0.0, y=0.0, z=10.0, code="ARBOL"),
    SurveyPoint(id="2", x=10.0, y=0.0, z=12.0, code="CERCA"),
))
group = NamedPointGroup(name="LEVANTAMIENTO_1", points=points, desc="Puntos de campo")

document = LandXMLDocument(surfaces=(surface,), point_groups=(group,), crs="EPSG:32618")

write_report = LandXMLWriter("salida.xml").write(document)
# confirmed real output: LandXMLWriteReport(surface_count=1, point_group_count=1,
#                          alignment_count=0, triangle_count=2, point_count=6, warnings=())

read_back = LandXMLReader("salida.xml").read()
# confirmed real output: 1 surface ("EXISTENTE"), 1 point group ("LEVANTAMIENTO_1"),
# crs == "EPSG:32618", tin.vertex_count == 4, both points with their own codes intact
```

## The "north east elev" coordinate convention — confirmed real, not just documented

The module's own docstring states LandXML coordinate text is always
"north east elev". Confirmed directly from the real file the example
above produces -- point `Point3D(10.0, 0.0, 12.0)` (x=10, y=0) was
written as `<P id="2">0.0000000000 10.0000000000 12.0000000000</P>`:
the first number (north = Y = 0) comes before the second (east = X =
10), the reverse of the `Point3D(x, y, z)` argument order. Getting
this backward is exactly the kind of silent, hard-to-notice bug this
convention exists to warn about -- confirmed here that TopoCore's own
reader/writer handle the swap correctly in both directions.

## Alignments — confirmed with a real 2-segment round trip

Horizontal alignment elements live in `topocore.alignment`, a
separate module from `topocore.io.landxml` itself -- fully covered in
[`alignment.md`](./alignment.md), including `ArcElement`/
`SpiralElement` and vertical profiles, this page only shows what's
needed to move an `Alignment` through LandXML itself.

```python
from topocore.geometry.point2d import Point2D
from topocore.alignment.elements import LineElement
from topocore.alignment.models import Alignment
from topocore.io.landxml import NamedAlignment

# Alignment.elements must chain continuously -- element i's own end
# must equal element i+1's own start; confirmed enforced at
# construction, not silently accepted if discontinuous.
segment_1 = LineElement(start=Point2D(0.0, 0.0), end=Point2D(100.0, 0.0))
segment_2 = LineElement(start=Point2D(100.0, 0.0), end=Point2D(150.0, 50.0))

alignment = Alignment(name="EJE_VIA_1", elements=(segment_1, segment_2), start_station=0.0)
named = NamedAlignment(name="EJE_VIA_1", alignment=alignment)

document = LandXMLDocument(alignments=(named,), crs="EPSG:32618")
LandXMLWriter("alineamiento.xml").write(document)
# confirmed real output: LandXMLWriteReport(alignment_count=1, ...)

read_back = LandXMLReader("alineamiento.xml").read()
# confirmed: 1 alignment, 2 elements, both preserved
```

`ArcElement`/`SpiralElement` also exist (curves and clothoid
spirals), confirmed present in `topocore.alignment.elements` -- see
[`alignment.md`](./alignment.md) for both, including a real,
confirmed finding about how `clockwise` selects between a minor and
major arc. Per this module's own documented scope, only `<Curve crvType="arc">` and
`<Spiral spiType="clothoid">` are supported when *reading* --
chord-defined curves, other spiral types, compound spirals, and
`<CircCurve>` vertical curves are skipped with an explicit warning in
`LandXMLReadReport`, not silently dropped or a hard failure.

## Error paths — confirmed real

```text
2 surfaces (or point groups, or alignments) sharing the same name
    -> LandXMLValidationError, raised at write() time, before any
       disk write -- confirmed: "Duplicated Surface name(s): [...]"

Reading a non-existent file
    -> LandXMLParseError, confirmed: wraps the underlying
       FileNotFoundError with a clear message naming the path
```

## Where to go next

- [`terrain.md`](./terrain.md) -- building the `TIN` a `NamedSurface`
  wraps.
- [`point-clouds.md`](./point-clouds.md) -- `SurveyPointSet`, used
  directly by `NamedPointGroup`.
- [`geodesy.md`](./geodesy.md) -- `detect_crs()`'s own LandXML-specific
  behavior: it reads only `<CoordinateSystem>`, with no `.prj`
  fallback, since `LandXMLReader.read()` returns a `LandXMLDocument`,
  not a `PointCloud`/`SurveyPointSet` -- an incompatible shape for the
  `.prj`-application machinery documented there.
