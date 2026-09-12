# `LandXMLReader`

```python
from topocore.io.landxml import LandXMLReader

reader = LandXMLReader("surface.xml")
document = reader.read()

# Or, to also get counts and warnings for skipped content:
document, report = reader.read_with_report()
```

## `read()` vs `read_with_report()`

`read()` returns only the `LandXMLDocument`. `read_with_report()`
returns the same document plus a `LandXMLReadReport` — counts of
surfaces/point groups/alignments/triangles/points actually read, and
a tuple of warning strings for anything recognized but skipped (an
unsupported surface type, an unsupported curve variant, a
point-by-reference `<CgPoint>`). `read()` is `read_with_report()`
with the report discarded — prefer `read_with_report()` whenever you
want to know if anything was silently skipped.

## What gets parsed

### `<Surfaces>` / `<Surface>`

Only `<Definition surfType="TIN">` is read. `GRID` and `ROAD` surface
types are recognized but skipped, with a warning — confirmed
directly:

```python
>>> doc, report = reader.read_with_report()
>>> report.warnings
("Surface 'GridSurf': surfType='GRID' is not supported (TIN only); skipped.",)
```

A `<Surface>` with no `<Definition>` at all is also skipped with a
warning rather than raising.

The TIN is built via `TIN.from_mesh()`, not `TIN.from_points()` — the
file's own `<Faces>` connectivity is preserved exactly as written,
never silently replaced by a fresh Delaunay triangulation of the
same points.

### `<CgPoints>`

Read as **direct children of the `<LandXML>` root** — a `<CgPoints>`
nested inside anything else is not found (this is standard LandXML
structure, not a TopoCore restriction).

An unnamed `<CgPoints>` (LandXML does not require the group-level
`name` attribute) is real, valid data — confirmed against a genuine
Autodesk Civil 3D 2007 export — and is given a deterministic
generated name, `Unnamed_CgPoints_{n}`, rather than being dropped.

A `<CgPoint pntRef="...">` (point-by-reference, permitted by the
schema) has no literal coordinate text to read and is skipped with a
warning:

```text
"Point group '<name>': <CgPoint pntRef='...'> (point-by-reference) is
not supported (no literal coordinates to read); skipped."
```

### `<Alignments>`

See [`alignments.md`](./alignments.md) for the full, precise scope
(which curve/spiral variants are read vs. skipped with a warning).

### `<Units>` and `<CoordinateSystem>`

See [`units-and-crs.md`](./units-and-crs.md).

## Errors

| Exception | When |
|---|---|
| `LandXMLParseError` | Not well-formed XML; a file that can't be opened; a required element/attribute in the *supported* subset is malformed (e.g. non-numeric coordinate text, an invalid `INF`/`-INF` radius) |
| `LandXMLValidationError` | The file violates a semantic invariant — wrong root element, duplicated surface/point-group/alignment names, a `<F>` face referencing an unknown point id, non-finite coordinates |

Confirmed directly:

```python
>>> LandXMLReader("not_xml_at_all.xml").read()
LandXMLParseError: '.../not_xml_at_all.xml' is not well-formed XML: no element found: line 1, column 19

>>> LandXMLReader("wrong_root.xml").read()   # <NotLandXML> instead of <LandXML>
LandXMLValidationError: Expected a <LandXML> root element, got <NotLandXML>.
```

Geometric consistency is also validated, not just structural
correctness — a `<Curve>` or `<Spiral>` whose declared radius/length
and its own Start/Center/End/PI coordinates are mutually
inconsistent (beyond `LANDXML_GEOMETRY_TOLERANCE`) raises
`LandXMLParseError`, naming the alignment:

```text
Alignment 'CL2': invalid <Spiral> geometry: SpiralElement geometry is
inconsistent: the chord length implied by radius/length (...) does
not match the actual distance between the zero-curvature point and
the finite-radius point (...).
```
