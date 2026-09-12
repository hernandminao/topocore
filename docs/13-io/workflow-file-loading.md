# `Workflow`: Loading a File — `read_point_cloud()` and `read_survey()`

`topocore.workflow.Workflow` is TopoCore's orchestrator: a mutable
object representing one execution, where every stage method returns
`self` for fluent chaining. Every pipeline starts with exactly one of
two **mutually exclusive root stages** — you must call one, and
calling either a second time (or calling the other afterward) is
rejected, not silently allowed.

```text
                    ┌──────────────────┐
                    │     Workflow()   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
     .read_point_cloud(path)         .read_survey(path)
              │                              │
              ▼                              ▼
     ArtifactType.POINT_CLOUD       ArtifactType.SURVEY_POINT_SET
              │                              │
              ▼                              ▼
   .classify_ground() / .build_tin()   .build_features_from_survey()
   / .classify_points() / ...          / .detect_features() / ...
```

This page documents these two entry points completely, with
every example run and verified against the real package — including
the exact error messages you'll see if something goes wrong.

## `read_point_cloud(path, **reader_kwargs)`

```python
from topocore.workflow import Workflow

wf = Workflow().read_point_cloud("survey.las")
```

**Dispatches by file extension only, case-insensitive.** This is a
complete, exhaustive table — there is no fallback, no content
inspection, and no other supported extension:

| Extension | Reader used |
|---|---|
| `.las` | `LASReader` |
| `.laz` | `LAZReader` |
| `.ply` | `PLYReader` |
| `.e57` | `E57Reader` |
| `.xyz` | `XYZReader` |
| `.csv` | `CSVReader` |
| `.pts` | `PTSReader` |

Any other extension (or no extension) raises immediately:

```python
>>> Workflow().read_point_cloud("data.desconocido")
WorkflowExecutionError: Stage read_point_cloud failed: Unsupported
point-cloud format: '.desconocido'. Supported formats: LAS, LAZ, PLY,
E57, XYZ, CSV, PTS.
```

`**reader_kwargs` are forwarded verbatim to whichever reader class
gets selected — e.g. `chunk_size=` (all formats), `has_header=`
(CSV only). One exception is handled for you: `E57Reader.chunk_size`
has no built-in default (confirmed elsewhere in this project's own
audit, unlike every other reader), so `read_point_cloud()` supplies
`chunk_size=1_000_000` automatically for `.e57` files if you don't
pass one — keeping this method's own contract uniform across every
format.

### Historical note: this dispatch used to be broken

Confirmed directly in the source: earlier, `read_point_cloud()` only
distinguished `.laz` from everything else, silently routing **every
other format** — PLY, E57, XYZ, CSV, PTS, and any unrecognized
extension — to `LASReader`. This is fixed now (the table above is
the corrected behavior), but it's worth knowing if you're looking at
older TopoCore-based code or documentation that assumed otherwise.

### Full worked examples, one per format

```python
from topocore.workflow import Workflow

Workflow().read_point_cloud("scan.las")
Workflow().read_point_cloud("scan.laz", chunk_size=500_000)
Workflow().read_point_cloud("mesh.ply")
Workflow().read_point_cloud("scan.e57")                  # chunk_size=1_000_000 applied automatically
Workflow().read_point_cloud("cloud.xyz")
Workflow().read_point_cloud("cloud.csv", has_header=True)
Workflow().read_point_cloud("cloud.pts")
```

**Every one of these produces a `PointCloud`** — none of them
preserve a text-valued survey code (`PointAttribute` is a closed,
entirely numeric enum). If your `.csv` (or `.xyz`/`.pts`) has a
leading point-ID column and/or a trailing field-code column (a
total-station/GNSS survey export, not a plain point-cloud export),
`read_point_cloud()` will silently misread it — see the worked,
verified example below and
[`csv-vs-survey.md`](./csv-vs-survey.md) for the full comparison.

### CRS detection happens automatically, inside the reader

`read_point_cloud()` does not call any separate CRS-detection step
of its own — each reader class (`LASReader`, `E57Reader`, etc.)
already resolves and sets `PointCloud.crs` internally, following the
native-then-`.prj` priority documented in
[`crs-detection.md`](./crs-detection.md). Confirmed directly: a
`.xyz` file with a correctly-named `.prj` sidecar in the same
directory produces a `PointCloud` with `.crs` already set, with no
extra call required:

```python
>>> wf = Workflow().read_point_cloud("survey.xyz")   # survey.prj sits alongside it
>>> wf.artifact(ArtifactType.POINT_CLOUD).crs
'EPSG:32617'
```

If no native CRS and no `.prj` sidecar apply, `.crs` is simply
`None` — never a guessed or default value.

### Errors, verified exactly as raised

```python
>>> Workflow().read_point_cloud("missing.xyz")
WorkflowExecutionError: Stage read_point_cloud failed: Point-cloud
file does not exist: 'missing.xyz'.

>>> Workflow().read_point_cloud("a_directory/")
WorkflowExecutionError: Stage read_point_cloud failed: Point-cloud
path is not a file: 'a_directory'.

>>> wf = Workflow().read_point_cloud("cloud.xyz")
>>> wf.read_point_cloud("other.xyz")   # second READ stage on the same Workflow
WorkflowStateError: Stage read_point_cloud requires artifacts to be
absent (point_cloud).
```

**Every exception from the underlying reader is wrapped in
`WorkflowExecutionError`**, with the original preserved via
`__cause__` — this is universal across all `Workflow` stages, not
specific to file reading (see `_execute_stage`'s own contract). A
malformed PLY header, a corrupted LAS file, an out-of-range ASCII
value — whatever the underlying reader would raise directly, calling
it through `Workflow` surfaces as `WorkflowExecutionError` wrapping
that original exception.

## `read_survey(path, **reader_kwargs)`

```python
from topocore.workflow import Workflow
from topocore.survey.formats import SurveyFormat

wf = Workflow().read_survey("levantamiento.csv", format=SurveyFormat.ID_XYZ_CODE)
```

Always uses `topocore.survey.SurveyTXTReader`, **regardless of file
extension** — confirmed directly: a file named `.txt`, `.csv`, or
with no extension at all is read identically, since this method
never inspects the extension. `**reader_kwargs` map directly onto
`SurveyTXTReader.__init__`'s own keyword arguments:
`format=`, `delimiter=`, `encoding=`.

This produces a `SurveyPointSet`, not a `PointCloud` — the one
container in TopoCore that preserves each point's text field code
(`SurveyPoint.code: str | None`), which is what
[`FeatureBuilder`](../11-catalogs/custom-catalogs.md) needs to turn
survey codes into `Feature`s.

### CRS detection

`SurveyTXTReader` detects a `.prj` sidecar the same way the ASCII
point-cloud readers do, but sets the result on `SurveyPointSet.crs`
as a real `CRS` object — not the `str | None` convention
`PointCloud.crs` uses. See
[`crs-detection.md`](./crs-detection.md) for why the two artifact
types carry different CRS value types.

### Errors, verified exactly as raised

```python
>>> Workflow().read_survey("no_header_no_format.csv")
WorkflowExecutionError: Stage read_survey failed: Could not detect a
header in 'no_header_no_format.csv' and no format preset was given.
Pass format=SurveyFormat.<...> explicitly.

>>> wf = Workflow().read_survey("levantamiento.csv", format=SurveyFormat.ID_XYZ_CODE)
>>> wf.read_point_cloud("cloud.xyz")   # the other root stage, same Workflow
WorkflowStateError: Stage read_point_cloud requires artifacts to be
absent (survey_point_set).
```

### A real pitfall: passing `format=` to a file that already has a header

If `format=` is given, header auto-detection is **never attempted at
all** — not even to skip the header row as non-data. The first row
is parsed as a data row regardless. Confirmed directly, this fails
loudly rather than silently:

```python
>>> # file starts with: "Punto,Este,Norte,Z,Descripcion"
>>> SurveyTXTReader("con_encabezado.csv", format=SurveyFormat.ID_XYZ_CODE).read()
SurveyRecordError: Invalid coordinate values at line 1 of the survey
file: ['Punto', 'Este', 'Norte', 'Z', 'Descripcion']
```

If your file has a real header, omit `format=` entirely and let
auto-detection handle it.

## `SurveyFormat` — every preset, documented completely

`SurveyTXTReader` resolves column order in this priority: **(1)** an
explicit `format=` preset, **(2)** auto-detected header, **(3)**
neither → `SurveyFormatError`. This section covers both paths in
full.

### Header auto-detection (no `format=` needed)

If the first line's fields match this table (case-insensitive,
spaces/underscores/dashes ignored), the header is detected and that
row is skipped as data. **Bilingual (ES/EN)** — this is TopoCore's
own primary use case (Latin American total-station exports), unlike
`topocore.io.ascii`'s own header table, which recognizes English
names only.

| Field | Recognized header names |
|---|---|
| `id` | `id`, `punto`, `pto`, `point`, `pointid`, `station` |
| `x` | `x`, `este`, `east`, `easting` |
| `y` | `y`, `norte`, `north`, `northing` |
| `z` | `z`, `cota`, `elevacion`, `elevation`, `elev`, `level`, `rl` |
| `code` | `code`, `codigo`, `descripcion`, `description`, `desc`, `obs`, `observacion`, `feature` |

Detection requires `x`, `y`, and `z` to all be found; `id` and `code`
are optional. Verified directly with the module's own documented
example:

```python
>>> # file: "Punto,Este,Norte,Z,Descripcion\n1,1815130.506,5041000.926,216.913,PT\n"
>>> SurveyTXTReader("archivo.csv").read().points[0]
SurveyPoint(id='1', x=1815130.506, y=5041000.926, z=216.913, code='PT')
```

### The 9 `SurveyFormat` presets (for headerless files)

| Preset | Column order | `id` | `code` |
|---|---|---|---|
| `ID_XYZ_CODE` | id, x, y, z, code | ✅ | ✅ |
| `NEZ` | **northing**, **easting**, z | — | — |
| `ENZ` | **easting**, **northing**, z | — | — |
| `PENZ` | id, easting, northing, z | ✅ | — |
| `PENZD` | id, easting, northing, z, description | ✅ | ✅ |
| `PNEZ` | id, **northing**, **easting**, z | ✅ | — |
| `PNEZD` | id, **northing**, **easting**, z, description | ✅ | ✅ |
| `XYZI` | x, y, z, intensity | — | — |
| `XYZRGB` | x, y, z, r, g, b | — | — |

**Why both NEZ/ENZ and PNEZ/PENZ exist as separate presets**: a
Northing-Easting-Z file and an Easting-Northing-Z file are
positionally identical in every way except which of the first two
columns is X and which is Y — there is no way to tell them apart by
inspecting the data itself. Naming the convention explicitly, rather
than asking the caller to work out `x_column`/`y_column` indices by
hand, turns a silent, undetectable mistake into one explicit choice
made once. Verified directly — the SAME file, read with `PENZ` vs.
`PNEZ`, produces swapped X/Y:

```python
>>> # file: "1,100.0,200.0,10.0\n"
>>> SurveyTXTReader(path, format=SurveyFormat.PENZ).read().points[0]
SurveyPoint(id='1', x=100.0, y=200.0, z=10.0, code=None)
>>> SurveyTXTReader(path, format=SurveyFormat.PNEZ).read().points[0]
SurveyPoint(id='1', x=200.0, y=100.0, z=10.0, code=None)
```

### `XYZI` / `XYZRGB`: recognized, but genuinely discarded

These two presets exist so files in these common point-cloud-style
layouts **parse without raising `SurveyFormatError`** rather than
being rejected outright — but the intensity/color columns are never
extracted anywhere. `SurveyPoint` has no field for them at all.
Verified directly:

```python
>>> # file: "1.0,2.0,3.0,150\n" (x, y, z, intensity)
>>> SurveyTXTReader(path, format=SurveyFormat.XYZI).read().points[0]
SurveyPoint(id='1', x=1.0, y=2.0, z=3.0, code=None)
```

The `150` intensity value is gone — not stored anywhere, not
recoverable from the returned `SurveyPointSet`. If you need
intensity or color preserved, use `read_point_cloud()` with a reader
that actually carries those attributes (LAS, LAZ, E57, PLY, or ASCII
with a recognized `intensity`/`red`/`green`/`blue` header) instead.

## `SurveyPointSet` — what you get back

```python
len(point_set)                  # point count
for point in point_set: ...     # iterate SurveyPoint
point_set[0]                    # index access
point_set.codes()               # frozenset[str] of every distinct code present
point_set.by_code("PT")         # tuple[SurveyPoint, ...] matching exactly that code
```

`SurveyPoint` itself: `id: str`, `x: float`, `y: float`, `z: float`,
`code: str | None`.

## Complete worked example: the exact pitfall this page exists to prevent

```python
from topocore.workflow import Workflow
from topocore.survey.formats import SurveyFormat

# A real total-station export: "id,easting,northing,elevation,code" with no header.
path = "BUZONES.csv"

# WRONG -- .csv always routes to CSVReader via read_point_cloud(),
# which has no column slot for the leading ID or the trailing code.
# The point number silently becomes X; the real X becomes Y; the
# real Y becomes Z; the real elevation and the code are dropped.
wrong = Workflow().read_point_cloud(path)

# CORRECT -- explicit format matching this file's real column layout.
correct = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
```

See [`csv-vs-survey.md`](./csv-vs-survey.md) for the full comparison
table and decision guide, and
[`../11-catalogs/custom-catalogs.md`](../11-catalogs/custom-catalogs.md)
for turning the resulting survey codes into `Feature`s via
`FeatureBuilder`.
