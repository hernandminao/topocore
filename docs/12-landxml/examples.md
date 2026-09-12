# Example: Read → Write Round Trip

The following was run end-to-end and its output captured directly —
nothing here is hand-written or assumed.

## Input file

```xml
<?xml version="1.0"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Units>
    <Metric linearUnit="meter"/>
  </Units>
  <CoordinateSystem name="EPSG:32617"/>
  <Surfaces>
    <Surface name="EG" desc="Existing Ground">
      <Definition surfType="TIN">
        <Pnts>
          <P id="1">0.0 0.0 10.0</P>
          <P id="2">0.0 10.0 12.0</P>
          <P id="3">10.0 0.0 11.0</P>
        </Pnts>
        <Faces>
          <F>1 2 3</F>
        </Faces>
      </Definition>
    </Surface>
  </Surfaces>
  <CgPoints name="ControlPoints" desc="Field control">
    <CgPoint name="CP1">0.0 0.0 10.0</CgPoint>
    <CgPoint name="CP2">5.0 5.0 10.5</CgPoint>
  </CgPoints>
  <Alignments>
    <Alignment name="CL1" desc="Centerline">
      <CoordGeom>
        <Line>
          <Start>0.0 0.0</Start>
          <End>100.0 0.0</End>
        </Line>
      </CoordGeom>
    </Alignment>
  </Alignments>
</LandXML>
```

Note: `<CgPoints>` must be a **direct child of `<LandXML>`**, not
nested inside anything else — this is standard LandXML structure.

## Reading it

```python
from topocore.io.landxml import LandXMLReader

document, report = LandXMLReader("sample.xml").read_with_report()
```

```python
>>> report.to_dict()
{'input_path': 'sample.xml', 'surface_count': 1, 'point_group_count': 1,
 'alignment_count': 1, 'triangle_count': 1, 'point_count': 5,
 'warnings': [], 'warning_count': 0}
```

`point_count` (5) is the sum of surface vertices (3) and point-group
points (2) — it counts points across every read section, not one
alone.

```python
>>> document.surfaces[0].name, document.surfaces[0].tin.triangle_count
('EG', 1)
>>> document.point_groups[0].name, len(document.point_groups[0].points.points)
('ControlPoints', 2)
>>> document.alignments[0].name
'CL1'
>>> document.crs
'EPSG:32617'
```

## Writing it back out

```python
from topocore.io.landxml import LandXMLWriter

write_report = LandXMLWriter("roundtrip.xml").write(document)
```

The resulting file (captured verbatim):

```xml
<?xml version='1.0' encoding='utf-8'?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Units>
    <Metric linearUnit="meter" />
  </Units>
  <CoordinateSystem name="EPSG:32617" />
  <Surfaces>
    <Surface name="EG" desc="Existing Ground">
      <Definition surfType="TIN">
        <Pnts>
          <P id="1">0.0000000000 0.0000000000 10.0000000000</P>
          <P id="2">0.0000000000 10.0000000000 12.0000000000</P>
          <P id="3">10.0000000000 0.0000000000 11.0000000000</P>
        </Pnts>
        <Faces>
          <F>1 2 3</F>
        </Faces>
      </Definition>
    </Surface>
  </Surfaces>
  <CgPoints name="ControlPoints" desc="Field control">
    <CgPoint name="CP1">0.0000000000 0.0000000000 10.0000000000</CgPoint>
    <CgPoint name="CP2">5.0000000000 5.0000000000 10.5000000000</CgPoint>
  </CgPoints>
  <Alignments>
    <Alignment name="CL1" length="100.0000000000" staStart="0.0000000000" desc="Centerline">
      <CoordGeom>
        <Line length="100.0000000000">
          <Start>0.0000000000 0.0000000000</Start>
          <End>100.0000000000 0.0000000000</End>
        </Line>
      </CoordGeom>
    </Alignment>
  </Alignments>
</LandXML>
```

Note the coordinate precision: every value is written to 10 decimal
places by default (see [`writer.md`](./writer.md) for why), and the
writer computes and adds a few derived attributes the input didn't
have (`length`, `staStart`) — these are legitimate LandXML
attributes describing the alignment's own geometry, not extra data
invented from nothing.

## Confirming the round trip

```python
>>> reread = LandXMLReader("roundtrip.xml").read()
>>> reread.surfaces[0].tin.vertex_count == document.surfaces[0].tin.vertex_count
True
>>> len(reread.point_groups[0].points.points) == len(document.point_groups[0].points.points)
True
>>> reread.alignments[0].name == document.alignments[0].name
True
>>> reread.crs == document.crs
True
```

## Handling a file with unsupported content

```python
document, report = LandXMLReader("civil3d_export.xml").read_with_report()

if report.warnings:
    for warning in report.warnings:
        print(warning)
```

A typical warning for a `<Surface surfType="GRID">`:

```text
Surface 'GridSurf': surfType='GRID' is not supported (TIN only); skipped.
```

See [`limitations.md`](./limitations.md) for the full list of what
produces a warning rather than being read.
