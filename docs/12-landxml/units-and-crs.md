# Units and CRS

Both are read and preserved for round-tripping by `LandXMLReader`
itself. Neither is automatically turned into a coordinate
*transformation* by anything in this package — but the CRS value
**can** be resolved into a real `CRS` object, via a separate function
covered later in this page.

## `<Units>` — `LinearUnit`

```python
class LinearUnit(StrEnum):
    METER = "meter"
    FOOT = "foot"
```

LandXML expresses the linear unit via a child element name —
`<Metric linearUnit="meter">` or `<Imperial linearUnit="foot">` —
not a single attribute. The reader resolves whichever is present
into this one enum (`<Imperial>`/`<Metric>` are mutually exclusive
per the schema; if a document has neither, `LinearUnit.METER` is the
default).

**Confirmed directly: `linear_unit` is metadata only.** Reading a
surface declared with `linearUnit="foot"` does **not** scale, divide,
or otherwise convert any coordinate value — a point written as
`328.084` (feet) is read back as the raw number `328.084`, not
converted to `100.0` (meters). `LandXMLDocument.linear_unit` tells
you what unit the *author* declared; TopoCore does not act on it.

If your workflow needs coordinates in a specific unit, converting
before or after LandXML IO is the caller's responsibility.

## `<CoordinateSystem>` — a raw string, not a `CRS`

```python
LandXMLDocument.crs: str | None
```

The reader looks for the first `<CoordinateSystem>` element and takes
whichever of `name`, `epsgCode`, or `desc` is present first, in that
order — treating all three identically as an opaque string:

```python
>>> doc.crs
'EPSG:32617'          # from <CoordinateSystem name="EPSG:32617"/>
>>> doc.crs
'2274'                # from <CoordinateSystem epsgCode="2274"/> (no `name`)
```

**This string, as returned by `LandXMLReader.read()` itself, is never
parsed, validated, or resolved into a `topocore.geodesy.CRS` object
by the reader.** No coordinate transformation is
ever applied based on it. It exists purely so a document's declared
CRS can be preserved through a read → write round trip without being
silently dropped.

This is a deliberate architectural boundary, not an oversight: per
the same rule that keeps `topocore.dxf.ExportContext.crs` separate
from `FeatureCollection`, LandXML's own CRS concept is kept at the
document level and is never injected into `TIN` or `SurveyPointSet`
themselves.

## Resolving it into a real `CRS` — a separate, dedicated function

A **separate** function, `topocore.io.landxml.crs_detection.detect_crs()`,
does perform this resolution — confirmed directly, it is genuinely
different code from `LandXMLReader` itself, not a later mode of it:

```python
>>> from topocore.io.landxml.crs_detection import detect_crs
>>> detect_crs("surface.xml")
CRS(EPSG:32617, 'WGS 84 / UTM zone 17N', Projected)
```

It re-reads the same `<CoordinateSystem>` value `LandXMLReader`
already extracts, and resolves it in priority order:

1. An `EPSG:NNNN`-prefixed string, or a bare integer (interpreted
   directly as an EPSG code) — from `name` or `epsgCode`.
2. As a last attempt, valid WKT.
3. Otherwise (e.g. a `desc`-only, free-text value like `"NAD83 State
   Plane Texas"`) — `None`, never a guess.

This same function is also what `topocore.io.crs.detect_crs()` (the
shared CRS-detection facade used across every TopoCore format) calls
for a `.xml` file — so code going through that facade already gets a
real `CRS`, without calling `LandXMLReader` directly at all. The
distinction that matters:

| | `LandXMLDocument.crs` (via `LandXMLReader.read()`) | `detect_crs()` (`crs_detection.py`, or the `topocore.io.crs` facade) |
|---|---|---|
| Type | `str \| None` | `CRS \| None` |
| Resolves EPSG/WKT | No | Yes |
| Used for coordinate transformation | Never | Only once you build a `CoordinateTransformer` from it yourself |

## Practical consequence

If you need TopoCore's own CRS/geodesy machinery (reprojection, unit
conversion, coordinate validation) applied to LandXML data: call
`detect_crs()` (directly, or via the `topocore.io.crs` facade) to get
a real `CRS`, then apply
[`geodesy.CRS`/`geodesy.CoordinateTransformer`](../14-geodesy/io-boundary.md) to
the `TIN`/`SurveyPointSet` coordinates yourself. Neither
`LandXMLReader` nor `detect_crs()` ever transforms a coordinate —
resolving *what* CRS a file declares and *applying* a transformation
based on it remain two separate, deliberate steps.
