# Limitations

Everything below is a deliberate scope boundary stated explicitly in
the source, not an unintentional gap. Each was confirmed by direct
execution during this documentation's own preparation, not merely
read off a comment.

## Not read or written at all

- **`<Feature>`** embedded inside a `<CgPoint>`. TopoCore has no
  domain model for it, and this subsystem does not invent one just
  to round-trip a LandXML element it can't otherwise use.
- **`<Surface surfType="GRID">` / `surfType="ROAD">`** — only `"TIN"`
  is supported. A GRID or ROAD surface is skipped with a warning,
  not read as an empty/wrong surface.
- **LandXML 1.0/0.88/2.0** — only the 1.2 namespace is recognized on
  read.
- **`<CgPoint pntRef="...">`** (point-by-reference) — only literal
  coordinate text is read.

## Alignment geometry — see `alignments.md` for full detail

- Chord-defined / three-point `<Curve>` (no `<Center>`).
- Any `<Curve crvType="...">` other than `"arc"`.
- Any `<Spiral spiType="...">` other than `"clothoid"` (cubic, bloss,
  biquadraticParabola, etc.).
- Curve-to-curve compound spirals (`<Spiral>` with both
  `radiusStart` and `radiusEnd` finite).
- Any `<ProfAlign>` point type other than `<PVI>` and `<ParaCurve>`
  (e.g. `<CircCurve>` vertical curves).

All of the above are skipped with an explicit warning in the read
report — never silently dropped with no trace, and never a hard
failure that blocks reading the rest of the file.

## Units and CRS — see `units-and-crs.md` for full detail

- **No unit conversion.** A coordinate declared in feet is read as
  the literal number written — never scaled to meters or any other
  unit.
- **`LandXMLReader`/`LandXMLDocument.crs` do not resolve CRS.**
  `LandXMLDocument.crs` is an opaque string (whichever of
  `name`/`epsgCode`/`desc` is present), never parsed into a
  `topocore.geodesy.CRS` object by the reader itself, and never used
  by it to transform coordinates. A **separate** function,
  `topocore.io.landxml.crs_detection.detect_crs()` (also reachable
  via the shared `topocore.io.crs.detect_crs()` facade), does resolve
  this same value into a real `CRS` — see
  [`units-and-crs.md`](./units-and-crs.md) for the full distinction.

## What round-trips and what doesn't

A file read then immediately written back out will **not** be
byte-identical to the original — attribute ordering, whitespace, and
any content in the unsupported categories above are not preserved
(they were never represented in the in-memory `LandXMLDocument` to
begin with). What's confirmed to round-trip correctly is the
*geometric and semantic content* of the supported subset: surface
TIN topology, point group membership and coordinates, alignment
horizontal/vertical element sequences, and the declared linear unit
and CRS string.

## Reporting, not silence

Every skip described above appends a human-readable warning to
`LandXMLReadReport.warnings` — `read()` discards these; use
`read_with_report()` if you need to know whether a file's full
content was represented, or only a subset of it.
