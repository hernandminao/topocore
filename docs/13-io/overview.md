# IO — Overview

TopoCore reads point cloud data from 4 format families into a single
internal representation, `PointCloud` (a sequence of `Chunk`s, each
holding `PointAttribute`-keyed arrays), and writes a subset of them
back out. LandXML is documented separately
([`../12-landxml/`](../12-landxml/)) because its own model isn't
just another `PointCloud` reader/writer — it carries surfaces,
control points, and alignments with their own document structure.

| Format | Read | Write | CRS detection | Required dependency |
|---|---|---|---|---|
| LAS | ✅ `LASReader` | ✅ `LASWriter`* | native (VLR) → `.prj` fallback | `laspy` |
| LAZ | ✅ `LAZReader` | ✅ `LAZWriter` | native (VLR) → `.prj` fallback | `laspy` + `lazrs` (compression backend) |
| E57 | ✅ `E57Reader` | — | `coordinateMetadata` → `.prj` fallback (native rarely populated in practice — see [`e57.md`](./e57.md)) | see [`e57.md`](./e57.md) |
| PLY | ✅ `PLYReader` | — | `.prj` only | none (stdlib only) |
| ASCII (XYZ) | ✅ `XYZReader` | ✅ `XYZWriter` | `.prj` only | none |
| ASCII (CSV) | ✅ `CSVReader` | — | `.prj` only | none |
| ASCII (PTS) | ✅ `PTSReader` | — | `.prj` only | none |

\* `LASWriter` exists and is fully functional, but the `las` package's
own docstring states only `LASReader` is considered public API — see
[`las-laz.md`](./las-laz.md) for what that means in practice.

`io/ascii/csv/writer.py` is deliberately excluded from this
inventory: it was established earlier in this project's own audit
that this file is not part of the current product surface.

## CRS detection — a separate concern from reading point data

Every format above can also carry (or acquire, via a `.prj`
sidecar) coordinate reference system information — this is fully
documented in [`crs-detection.md`](./crs-detection.md), including
the shared `detect_crs()` facade, the native-vs-`.prj` priority for
each format, and how a detected CRS propagates into `PointCloud.crs`.
Survey TXT (a 9th, non-point-cloud entry point covered in
[`csv-vs-survey.md`](./csv-vs-survey.md)) and LandXML (documented
separately in [`../12-landxml/`](../12-landxml/)) each have their own
CRS path too — the real total across everything this section and
[`../12-landxml/`](../12-landxml/) cover is these 7 point-cloud
formats plus Survey TXT plus LandXML, not a round "7 formats" figure.

## Common shape

Every reader in this section converges on the same two protocols:

```text
PointCloudReader                    PointCloudWriter
      │                                    │
      ├── __iter__() -> Iterator[Chunk]    ├── write(cloud: PointCloud)
      └── close()                          └── close()
```

A reader is iterated chunk by chunk — never loading an entire file
into memory at once — and each `Chunk` carries whichever
`PointAttribute`s that block of the source file actually had (not
every reader/format produces the same attribute set; see each
format's own page for exactly which attributes it maps).

## In this section

- [`las-laz.md`](./las-laz.md) — LAS and LAZ, including a real,
  historically-fixed precision defect worth knowing about
  (now corrected), a confirmed current limitation (`header` is
  always `None`), and native CRS detection.
- [`e57.md`](./e57.md) — the ASTM E57 reader, including why its own
  native CRS detection is far less reliable in practice than LAS's.
- [`ply.md`](./ply.md) — the PLY reader, built on the header parser
  already audited during coverage hardening.
- [`ascii.md`](./ascii.md) — XYZ, CSV, and PTS, and the shared
  parsing/mapping infrastructure behind all three.
- [`crs-detection.md`](./crs-detection.md) — the complete CRS
  detection architecture across every format: the shared
  `detect_crs()` facade, native-vs-`.prj` priority per format, and
  how detection differs from declaration, transformation, and
  georeferencing.
- [`csv-vs-survey.md`](./csv-vs-survey.md) — **read this before
  choosing a reader for a `.csv` file.** `CSVReader` and
  `SurveyTXTReader` both accept CSV, but TopoCore does not detect
  which one your file needs — picking wrong silently produces
  incorrect geometry.
- [`workflow-file-loading.md`](./workflow-file-loading.md) — the
  complete, verified reference for `Workflow.read_point_cloud()` and
  `Workflow.read_survey()`, including every `SurveyFormat` preset.
- [`formats-and-contracts.md`](./formats-and-contracts.md) — the
  shared `PointCloudReader`/`PointCloudWriter` contract, the
  `PointAttribute` mapping table across every format, and error
  handling conventions common to all of them.
