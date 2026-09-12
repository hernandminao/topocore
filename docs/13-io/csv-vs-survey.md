# CSV vs. Survey Text: Choosing the Right Entry Point

A `.csv` file can mean two genuinely different things in TopoCore,
and **TopoCore does not detect which one you have — you must choose
explicitly.** Getting this wrong produces silently incorrect
geometry, with no error and no warning. This page exists because
that failure mode is real and easy to hit in practice (confirmed
directly against a real-world Latin American manhole survey export).

> For the complete `Workflow`-level reference — every example run
> and verified, every `SurveyFormat` preset documented, and the full
> error-message catalogue for both entry points — see
> [`workflow-file-loading.md`](./workflow-file-loading.md). This page
> focuses on the comparison and the decision; that one is the
> exhaustive reference.

## The two readers

| | `CSVReader` (`topocore.io.ascii.csv`) | `SurveyTXTReader` (`topocore.survey.reader`) |
|---|---|---|
| Produces | `PointCloud` | `SurveyPointSet` |
| Assumes | A point-cloud export: `x,y,z[,attr...]` | A total-station/GNSS survey file: `id,x,y,z,code` (or similar) |
| Header detection | English column names only (`x`, `y`, `z`, `intensity`, ...) | **Bilingual (ES/EN)**: `punto`/`point`, `este`/`east`, `norte`/`north`, `cota`/`elevation`, `codigo`/`code`, ... |
| Headerless file | Assumes columns 0/1/2 are x/y/z — **wrong if the file has a leading ID column** | Requires an explicit `format=SurveyFormat.<preset>` — refuses to guess |
| Survey code / description column | No slot for it — silently dropped | Preserved as `SurveyPoint.code: str \| None` |
| CRS detection | `.prj` sidecar, via `topocore.io.crs.detect_crs()` — result is `str \| None` on `PointCloud.crs` | `.prj` sidecar, via a direct `ExternalCRSDetector` call (not the `detect_crs()` facade) — result is `CRS \| None` on `SurveyPointSet.crs` |
| On ambiguity | Silently uses wrong columns, no error | Raises `SurveyFormatError` explicitly |

Both readers can detect a `.prj` sidecar identically named next to
the data file — but the *type* they set it as differs
(`PointCloud.crs` is a string; `SurveyPointSet.crs` is a real `CRS`
object), and `SurveyTXTReader` never goes through the shared
`detect_crs()` facade at all (its own `.txt`/no-extension files are
too generic a signal for that facade's extension-based dispatch to
handle safely). See [`crs-detection.md`](./crs-detection.md) for the
full architecture.

`SurveyTXTReader` exists specifically because `ASCIIParser`
(`CSVReader`'s own foundation) has no column slot for a point ID or a
field code in the headerless case — confirmed directly in the
reader's own module docstring. It is a separate, purpose-built
parser, not a variant of the ASCII pipeline.

## Dispatch is by explicit method call, not automatic

At the `Workflow` level, this shows up as two separate, mutually
exclusive stage methods:

```python
Workflow().read_point_cloud(path)   # ALWAYS routes .csv to CSVReader
Workflow().read_survey(path)        # ALWAYS uses SurveyTXTReader, any extension
```

**`read_point_cloud(path)` dispatches by file extension only** —
`.las`→`LASReader`, `.laz`→`LAZReader`, `.ply`→`PLYReader`,
`.e57`→`E57Reader`, `.xyz`→`XYZReader`, `.csv`→`CSVReader`,
`.pts`→`PTSReader`. It never inspects the file's actual column
layout. Confirmed directly: calling
`Workflow().read_point_cloud("survey_with_id_and_code.csv")` produces
the exact same wrong-column result as calling `CSVReader` directly —
the point ID becomes X, the real X becomes Y, and so on. Routing
through the high-level `Workflow` API does **not** protect you from
this.

**`read_survey(path)` ignores the file extension entirely** and
always uses `SurveyTXTReader`. Confirmed directly: it reads a survey
file correctly whether it's named `.csv`, `.txt`, or has no
extension at all — extension plays no role in this method's own
dispatch.

## How to decide which one to call

Ask: **does this file have a leading ID column and/or a trailing
code/description column?**

- **No** (just `x,y,z[,attr...]`, or a recognized point-cloud header
  like `x,y,z,intensity`) → `read_point_cloud()` / `CSVReader`.
- **Yes** (a total-station or GNSS export: point number, coordinates,
  and a field code like `PT`, `BV`, `POSTE`) → `read_survey()` /
  `SurveyTXTReader`, with an explicit `format=` if the file has no
  header row.

### `SurveyFormat` presets (for headerless survey files)

| Preset | Column order |
|---|---|
| `ID_XYZ_CODE` | id, x, y, z, code |
| `NEZ` | northing, easting, z |
| `ENZ` | easting, northing, z |
| `PENZ` | id, easting, northing, z |
| `PENZD` | id, easting, northing, z, description |
| `PNEZ` | id, northing, easting, z |
| `PNEZD` | id, northing, easting, z, description |
| `XYZI` | x, y, z, intensity |
| `XYZRGB` | x, y, z, r, g, b |

If your file *does* have a header row, `SurveyTXTReader` detects it
automatically (bilingual ES/EN column names) and no `format=` is
needed at all.

### Worked example

A real manhole survey file, `id,x,y,z,code` layout, no header row:

```python
from topocore.survey.reader import SurveyTXTReader
from topocore.survey.formats import SurveyFormat

points = SurveyTXTReader("survey.csv", format=SurveyFormat.ID_XYZ_CODE).read()
# points.points[0].code -> "PT", "BV", "BZ12", etc. -- preserved correctly
```

Calling `CSVReader("survey.csv").read()` on the exact same file
instead would read the point-number column as `X`, the real
Easting as `Y`, the real Northing as `Z`, and silently drop both the
real elevation and the survey code.

## Feeding survey codes into `Feature`/DXF

Only `SurveyPointSet` (via `read_survey()`) carries the survey code
through to `FeatureBuilder`. See
[`../11-catalogs/custom-catalogs.md`](../11-catalogs/custom-catalogs.md)
for registering your own field codes, and
[`../17-features/`](../17-features/) for how `FeatureBuilder` turns
a code + a run of points into a `Feature`. `PointCloud` (from
`read_point_cloud()`) has **no attribute slot for a text code at
all** — `PointAttribute` is a closed, entirely numeric enum (see
[`formats-and-contracts.md`](./formats-and-contracts.md)) — so this
path is a dead end for survey-code-driven feature extraction,
regardless of which point-cloud reader produced it.
