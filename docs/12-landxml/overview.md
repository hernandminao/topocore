# LandXML — Overview

TopoCore reads and writes **LandXML 1.2** files: the interchange
format used by Civil 3D, Trimble Business Center (TBC), and similar
civil engineering software to carry surfaces, control points, and
road/rail alignments between applications.

Support is **read and write, symmetric**, for exactly three LandXML
element families:

| LandXML element | TopoCore model | Direction |
|---|---|---|
| `<Surfaces>` (TIN definitions) | `NamedSurface` (wraps `terrain.TIN`) | read + write |
| `<CgPoints>` (control point groups) | `NamedPointGroup` (wraps `survey.SurveyPointSet`) | read + write |
| `<Alignments>` (`<CoordGeom>` + `<Profile>`/`<ProfAlign>`) | `NamedAlignment` (wraps `alignment.Alignment`/`DesignProfile`) | read + write |

`<Feature>` embedded inside a `<CgPoint>` is explicitly **out of
scope** — TopoCore has no domain model for it, and this subsystem
does not invent one just to round-trip a LandXML element it can't
otherwise use.

## What "supported" means here

This documentation only describes behavior confirmed by direct
execution against hand-built and real-world LandXML files during
this project's own audit work — not everything the LandXML 1.2
schema technically permits. Where the schema allows more than
TopoCore reads, that's stated explicitly as a limitation, not left
implicit.

## Pipeline

```text
LandXML file
      │
      ▼
LandXMLReader.read() / .read_with_report()
      │
      ├── validate_xml()          -- structural: root element, required attrs
      ├── _read_surfaces()        -- TIN.from_mesh(), preserves file's own <Faces>
      ├── _read_point_groups()    -- SurveyPointSet, file order preserved
      ├── _read_alignments()      -- Line/Curve(arc)/Spiral(clothoid) + Profile
      ├── _read_units_and_crs()   -- LinearUnit + raw CRS string
      └── validate_document()     -- semantic: duplicate names, non-finite coords
      │
      ▼
   LandXMLDocument
      │
      ▼
LandXMLWriter.write()  ──►  LandXML 1.2 file
```

`LandXMLDocument` is a plain, frozen container — it does not extend
`TIN`, `SurveyPointSet`, or `Alignment` with a `name` field. Instead,
`NamedSurface`/`NamedPointGroup`/`NamedAlignment` wrap the existing
domain object alongside the `name`/`desc` metadata LandXML itself
carries, keeping the format's own naming concern out of TopoCore's
core geometry types.

## In this section

- [`reader.md`](./reader.md) — `LandXMLReader`, what it parses, and
  every confirmed error/warning path.
- [`writer.md`](./writer.md) — `LandXMLWriter` and the write-side
  validation contract.
- [`coordinate-convention.md`](./coordinate-convention.md) — the
  "north east elev" convention and why it matters.
- [`alignments.md`](./alignments.md) — exactly which horizontal and
  vertical alignment geometry is supported.
- [`units-and-crs.md`](./units-and-crs.md) — what "CRS support"
  actually means here (spoiler: less than a resolved `CRS` object).
- [`limitations.md`](./limitations.md) — a consolidated list of
  everything the LandXML schema allows that TopoCore does not read
  or write.
- [`examples.md`](./examples.md) — a verified, working end-to-end
  read → write → read round trip.
