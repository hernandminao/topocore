# Coordinate Convention — "north east elev"

This is the single most dangerous silent mistake when working with
LandXML: coordinate text is **not** "x y z".

LandXML encodes a point as whitespace-separated text inside an
element:

```xml
<P id="1">775431.17 2603215.79 62.9</P>
```

This is always **Northing Easting Elevation** order — confirmed both
by the LandXML 1.2 schema documentation and by Autodesk's own Civil
3D LandXML support notes ("All coordinate locations are always
treated as Northing, Easting, Elevation, or Y, X, Z").

## The mapping TopoCore applies

```text
LandXML text:  "north east elev"   (or "north east" for 2D)
                      │
                      ▼
Point3D:        x = east, y = north, z = elev
```

This keeps `Point3D.x`/`SurveyPoint.x` meaning Easting and `.y`
meaning Northing — consistent with the convention already
established elsewhere in `topocore.survey.models.SurveyPoint`.

This mapping lives in exactly one place,
`topocore.io.landxml.coordinates`, and both `reader.py` and
`writer.py` call into it — the convention is applied once, so reading
and writing can never silently drift apart from each other.

## 2D points

A point with only two values ("north east", no elevation) is valid
LandXML — plan-only control points commonly omit elevation. TopoCore
accepts this and defaults `z=0.0`. This is a deliberate, documented
default, not a silent failure: `Point3D`/`SurveyPoint` have no
optional-z variant, so *some* value has to be chosen, and `0.0` is
it.

## What raises an error

```python
>>> parse_point_text("")
LandXMLParseError: Point has no coordinate text -- likely defined via
a 'pntRef' attribute referencing another point, which this reader
does not support (only literal 'north east [elev]' text).

>>> parse_point_text("1.0 2.0 3.0 4.0")   # 4 values
LandXMLParseError: Expected 2 or 3 coordinate values in 'north east
[elev]' order, got 4: '1.0 2.0 3.0 4.0'

>>> parse_point_text("1.0 not_a_number")
LandXMLParseError: Non-numeric coordinate value in '1.0 not_a_number'
```

## Alignments use the same convention, differently expressed

`<CoordGeom>` elements (`<Start>`, `<End>`, `<Center>`, `<PI>`) use
the same "north east" 2D convention — see
[`alignments.md`](./alignments.md). `<PVI>`/`<ParaCurve>` vertical
profile points do **not**: they're "station elevation", an
unambiguous pair with no Northing/Easting swap to worry about, since
station and elevation are unrelated axes.
