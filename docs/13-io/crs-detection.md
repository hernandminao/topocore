# CRS Detection

This page documents the complete CRS-detection architecture across
every format TopoCore reads. It does not cover coordinate
transformation or control-point georeferencing — see the note on
terminology below; those are different operations, documented
separately in [`../14-geodesy/`](../14-geodesy/).

## Four different operations — do not conflate them

TopoCore's own code and this documentation distinguish four
genuinely separate operations. A file's coordinates can go through
any subset of these, independently:

| Operation | Question it answers | Where |
|---|---|---|
| **Detection** | "What CRS does this file's own header/sidecar declare?" | `detect_crs(path) -> CRS \| None` (this page) |
| **Declaration / propagation** | "What CRS is this in-memory object *currently* expressed in?" | `artifact.crs`, set by a reader, carried forward by later stages |
| **Transformation** | "Convert this artifact's coordinates from a known source CRS to a known target CRS." | `topocore.geodesy.CoordinateTransformer`, `Workflow.transform_crs()` |
| **Georeferencing** | "Estimate a transformation from local/arbitrary coordinates to a real CRS, using known control-point correspondences." | `topocore.geodesy.georeferencing`, `Workflow.georeference()` |

Detection never transforms anything — it only reads what a file
already says about itself. Declaration/propagation never resolves or
validates anything — it is just data flowing from a reader into an
artifact's `.crs` field, and from one stage's output to the next
stage's input. Transformation requires **both** a source and a
target CRS already known — it never guesses either one. Georeferencing
is the only one of the four that produces a *new* transformation
rather than consuming an existing one — it estimates the transformation
itself, from `ControlPoint` correspondences, rather than being told
what it is; see
[`../14-geodesy/georeferencing.md`](../14-geodesy/georeferencing.md).

## The detection decision tree

```text
                    detect_crs(path)
                         |
             +-----------+-----------+
             |                       |
       native CRS present     no native CRS
       (format-specific)     (or format has none)
             |                       |
             v                       v
       native CRS            ExternalCRSDetector
       (returned as-is,            |
        .prj never consulted)      v
                                  .prj sidecar
                                    |
                       +------------+------------+
                       |                         |
                  present & valid WKT      absent, empty, or
                       |                    unparseable
                       v                         v
                      CRS                      None
```

A native CRS, once found, is **never** compared against or
overridden by a `.prj` sidecar — even one declaring a genuinely
different CRS. The `.prj` fallback path is only ever consulted when
the native strategy itself returns `None`.

## Per-format strategy

| Format | Native CRS source | `.prj` fallback | Entry point |
|---|---|---|---|
| LAS | Embedded VLR (GeoTIFF keys or WKT) | Yes, if native is absent | `detect_crs()` facade |
| LAZ | Same as LAS (compression is orthogonal) | Yes, if native is absent | `detect_crs()` facade |
| E57 | `coordinateMetadata` (free-text field) | Yes, if native is absent | `detect_crs()` facade |
| LandXML | `<CoordinateSystem>` (`name`/`epsgCode`/`desc`) | **No** — see below | `detect_crs()` facade |
| PLY | none | Yes — only mechanism | `detect_crs()` facade |
| XYZ | none | Yes — only mechanism | `detect_crs()` facade |
| CSV | none | Yes — only mechanism | `detect_crs()` facade |
| PTS | none | Yes — only mechanism | `detect_crs()` facade |
| Survey TXT | none | Yes — only mechanism | **Not** the facade — see below |

This is 9 real entry points across the formats TopoCore reads, not a
round "7 formats" figure — LAS and LAZ share one native mechanism but
are still 2 separate readers/extensions, and Survey TXT and LandXML
each have their own distinct CRS path alongside the 7 point-cloud
formats.

### LAS / LAZ

`topocore.io.las.crs_detection.detect_crs()` reads whatever CRS the
file's own header already declares, via `laspy`'s
`LasHeader.parse_crs()` — either the GeoTIFF-keys VLR path (LAS < 1.4
/ point format < 6) or the WKT VLR path (LAS >= 1.4 / point format
>= 6). Confirmed directly, including for a genuinely non-EPSG,
custom-parameter CRS (WKT path only — `laspy`'s own GeoTIFF-keys
writer refuses a projected CRS with no EPSG code) and a compound
horizontal+vertical CRS (`EPSG:32618+5773`; `.epsg` is correctly
`None` for the returned `CRS` in that case — inherent to
`pyproj.CRS.to_epsg()` for a compound CRS, not a limitation
introduced here).

No CRS present, or a structurally broken CRS VLR, is treated
identically to "no CRS" — never an error. A missing *file*
(`FileNotFoundError`) is a separate, genuine caller error, not
caught here.

### E57 — native detection is real, but unreliable in practice

`topocore.io.e57.crs_detection.detect_crs()` reads the file's own
`/coordinateMetadata` field (a file-level field, confirmed present
at most once, never per-scan) and attempts to parse it as WKT.

**This is genuinely less reliable than LAS's own native detection —
confirmed from 4 independent sources, not assumed:**

- The ASTM E57 spec itself only describes `coordinateMetadata` as an
  *optional* string, with no enforced structure.
- MATLAB's own official `readCRS()` documentation example, run
  against MathWorks' own sample E57 dataset, reports "No CRS data
  available."
- A real-world E57 writer (the CloudCompare/`PCD-E57` converter)
  writes a literal empty string by default.
- `pye57` itself locks `coordinateMetadata` to an empty string the
  moment a file is opened for writing via its own high-level `E57`
  class — populating real WKT requires its low-level `ImageFile` API,
  which most real-world writers do not use.

**In practice, `detect_crs()` returning `None` for a real-world E57
file is the overwhelmingly common outcome** — not because detection
itself is broken, but because most E57 files in the wild simply never
had a CRS written into them. Do not treat E57 CRS detection as
comparably reliable to LAS's own.

### LandXML — native only, deliberately no `.prj` fallback

`topocore.io.landxml.crs_detection.detect_crs()` resolves
`<CoordinateSystem>`'s `name`/`epsgCode`/`desc` (in that priority) into
a real `CRS` — an `EPSG:NNNN`-prefixed string or bare integer as an
EPSG code, or valid WKT as a last attempt. See
[`../12-landxml/units-and-crs.md`](../12-landxml/units-and-crs.md)
for the full detail, including the distinction between this function
and `LandXMLDocument.crs` itself (a raw, unresolved string).

**No `.prj` fallback exists for LandXML.** `LandXMLReader.read()`
returns a `LandXMLDocument`, not a `PointCloud`/`SurveyPointSet` — an
incompatible shape for the `.prj`-application machinery
(`apply_external_crs()`/`apply_crs_with_native_priority()`, both of
which require a `.crs` setter that `LandXMLDocument` does not have).
This is a deliberate scope boundary of the current implementation,
not an oversight — adding it would require a new mechanism, not a
one-line change.

### PLY / XYZ / CSV / PTS — `.prj` is the only mechanism

None of these 4 formats has any internal CRS concept of its own —
confirmed directly across each format's own reader. `ExternalCRSDetector`
(`topocore.io.crs.external`) is the single, shared strategy for all
4: look for a sidecar file at the same path with the main file's
extension replaced by `.prj` (`survey.xyz` -> `survey.prj`, never
`survey.xyz.prj`), read it, and attempt to parse its content as WKT.

`.prj` is not a TopoCore invention — it is the decades-old ESRI
Shapefile sidecar convention (`.shp`/`.shx`/`.dbf`/`.prj`), still the
de facto standard across the wider GIS ecosystem for exactly this
purpose. Sidecar files are read with `utf-8-sig` (not plain `utf-8`)
specifically to correctly handle a leading UTF-8 byte-order-mark,
common in `.prj` files produced by Windows/ESRI tooling.

`None` is returned, never an error, for: no `.prj` file present; an
empty or whitespace-only `.prj` file; a `.prj` file containing
anything other than the CRS's own WKT verbatim (a comment, leading
blank lines, extra metadata — the whole string must be valid WKT, no
best-effort extraction is attempted); or a `.prj` that exists but
cannot be read (treated the same as absent).

### Survey TXT — its own direct call, not the `detect_crs()` facade

`topocore.survey.reader.SurveyTXTReader` calls
`ExternalCRSDetector().detect(self._path)` directly, setting the
result on `SurveyPointSet.crs` — it does **not** go through
`topocore.io.crs.detect_crs()`. This is deliberate: the facade
dispatches purely by file extension, and `.txt` (or no extension at
all — a survey file's extension is not reserved or distinctive) is
far too generic a signal to route automatically. If you already know
a path is a survey text file, `SurveyTXTReader` handles its own `.prj`
detection internally; there is no separate call you need to make.

**Type note**: `SurveyPointSet.crs` is a real `CRS` object (matching
what `ExternalCRSDetector().detect()` returns directly), unlike
`PointCloud.crs`/`FeatureCollection.crs`/`LandXMLDocument.crs`, which
are all `str | None` (the `f"EPSG:{code}"`-or-name convention). See
each artifact's own model docstring for why.

## The `detect_crs()` facade

```python
from topocore.io.crs import detect_crs

crs = detect_crs("survey.las")   # CRS | None
```

A single, format-agnostic entry point covering LAS, LAZ, E57,
LandXML (`.xml`), PLY, XYZ, CSV, and PTS — dispatching by file
extension (case-insensitive) to whichever per-format strategy above
applies. Survey TXT is intentionally excluded (see above).

`FileNotFoundError` is raised if `path` does not exist, checked
unconditionally before any extension-based dispatch — so a missing
file is never confused with "an unrecognized extension." An
unrecognized extension itself resolves to `None`, not an error — a
softer failure mode than `Workflow.read_point_cloud()`'s own
extension dispatch (which does reject an unsupported format outright),
since this facade's own job is purely informational and was never
meant to gate whether a file can be processed further.

**Header-only, confirmed with a real spy, regardless of file size.**
Point/scan data is never read by this function for any format —
confirmed directly via monkeypatched spies on the actual data-reading
entry points (`laspy.LasReader.chunk_iterator`, `pye57`'s own
scan-reading calls). The main data file is opened exactly once; the
`.prj` fallback (when it applies) opens only the separate sidecar
file, never re-opening the main file.

## Propagation into artifacts

Once detected, a CRS is set on the resulting artifact by the reader
itself (or, for LAS/E57, by `apply_crs_with_native_priority()`):

| Artifact | Field | Type | Set by |
|---|---|---|---|
| `PointCloud` | `.crs` | `str \| None` | Each point-cloud reader, via `apply_crs_with_native_priority()`/`apply_external_crs()` |
| `SurveyPointSet` | `.crs` | `CRS \| None` | `SurveyTXTReader`, directly |
| `LandXMLDocument` | `.crs` | `str \| None` | `LandXMLReader`, as a **raw, unresolved** string — see [`../12-landxml/units-and-crs.md`](../12-landxml/units-and-crs.md) |
| `FeatureCollection` | `.crs` | `str \| None` | `Workflow.detect_features()`/`build_features_from_survey()`, copied or converted from the source artifact |

From there, propagation through the rest of a `Workflow` pipeline
(`transform_crs()`, `transform_vertical()`, `resolve_sides()`,
exports) is covered in [`../14-geodesy/`](../14-geodesy/).

## What this layer does not do

- It does not transform coordinates. A detected CRS is metadata about
  what the coordinates currently *are* — using it to convert to a
  different CRS is a separate, explicit step
  (`CoordinateTransformer`/`Workflow.transform_crs()`).
- It does not resolve conflicting CRS declarations from multiple
  sources for you — there is no `CRSStatus.CONFLICT` mechanism in the
  current implementation.
- It does not georeference local/arbitrary coordinates. That is a
  distinct capability requiring control-point correspondences — see
  [`../14-geodesy/georeferencing.md`](../14-geodesy/georeferencing.md).
- It never invents a CRS. Every path above that cannot determine one
  returns `None`, not a default or a guess.
