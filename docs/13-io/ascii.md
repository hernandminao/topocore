# ASCII (XYZ / CSV / PTS)

> **Before using `CSVReader` on a survey-style file (with a point ID
> and/or a field code column), read
> [`csv-vs-survey.md`](./csv-vs-survey.md) first.** `CSVReader` will
> silently misread that layout — a leading ID column becomes X, the
> real coordinates shift over, and the field code is dropped.
> `topocore.survey.reader.SurveyTXTReader` is the correct reader for
> that case, and TopoCore does not choose between the two for you.

```text
XYZ / CSV / PTS file
        │
        ▼
   ASCIIParser  (delimiter/header detection, column buffering)
        │
        ▼
   ASCIIRecordBatch
        │
        ▼
   ASCIIConverter
        │
        ▼
   Chunk  (X, Y, Z + any recognized scalar/vector column)
        │
        ▼
   PointCloud
```

All three formats share the same underlying `ASCIIParser` and
`ASCIIConverter` — only the delimiter, header-detection default, and
(for PTS) an extra leading point-count line differ between them.

**Note on `ColumnMapper`**: `topocore.io.ascii.mapper.ColumnMapper`
(documented separately, including its own known collision finding,
`COLUMN-MAPPER-001`) is **not** used anywhere in this pipeline —
confirmed directly via search. The live path (`ASCIIParser` →
`ASCIIConverter`) uses `ASCIIConverter`'s own, separate
`ATTRIBUTE_MAPPING` table (below). Do not conflate the two.

## `XYZReader`

```python
from topocore.io.ascii.xyz.reader import XYZReader

reader = XYZReader("cloud.xyz", chunk_size=1_000_000, encoding="utf-8")
```

Delimiter and header presence are **both auto-detected** — nothing
to configure. Confirmed directly:

- A file with a plain `x y z` triplet per line (no header) is read
  correctly, columns 0/1/2 defaulting to X/Y/Z.
- A file whose first line is a recognizable header (`x y z
  intensity`) is read with that header respected — `intensity`
  correctly resolves to `PointAttribute.INTENSITY`, not treated as a
  4th coordinate.

## `CSVReader`

```python
from topocore.io.ascii.csv.reader import CSVReader

reader = CSVReader("cloud.csv", has_header=None)
```

Delimiter is **fixed to `,`** (not auto-detected, unlike XYZ).
`has_header` can be forced explicitly (`True`/`False`) or left `None`
for auto-detection, same as XYZ.

## `PTSReader`

```python
from topocore.io.ascii.pts.reader import PTSReader

reader = PTSReader("cloud.pts")
```

Structurally different from the other two: a PTS file's **first
line is a declared point count**, and the reader verifies the actual
number of points read matches it:

```python
>>> list(PTSReader("empty.pts"))
CorruptedFileError: PTS file is missing its point count.

>>> list(PTSReader("bad_count.pts"))   # first line is "abc"
CorruptedFileError: PTS point count must be an integer.

>>> list(PTSReader("negative.pts"))    # first line is "-1"
CorruptedFileError: PTS point count cannot be negative.

>>> list(PTSReader("mismatch.pts"))    # declares 3, file has 2 data lines
CorruptedFileError: PTS point count mismatch: header declares 3, read 2 points.
```

PTS always treats the remaining lines as headerless, delimiter-
auto-detected data (`has_header=False` is fixed, not auto-detected —
the point-count line already establishes the file's own convention).

## Attribute mapping (shared by all 3 readers)

| Column name | `PointAttribute` |
|---|---|
| `x`, `y`, `z` | `X`, `Y`, `Z` |
| `intensity` | `INTENSITY` |
| `classification` | `CLASSIFICATION` |
| `return_number` | `RETURN_NUMBER` |
| `number_of_returns` | `NUMBER_OF_RETURNS` |
| `scan_angle` | `SCAN_ANGLE` |
| `gps_time` | `GPS_TIME` |
| `nir` | `NIR` |
| `user_data` | `USER_DATA` |
| `point_source_id` | `POINT_SOURCE_ID` |
| `red`, `green`, `blue` (all 3 present) | `COLOR` (combined) |
| `nx`, `ny`, `nz` (all 3 present) | `NORMAL` (combined) |

An unrecognized column name is simply not carried into the resulting
`Chunk` — it's dropped, not an error.

## CRS detection

None of the 3 ASCII formats has any internal CRS concept — a `.prj`
sidecar is the only mechanism, shared identically across XYZ, CSV,
and PTS via `ExternalCRSDetector`. See
[`crs-detection.md`](./crs-detection.md) for the full detail,
including exactly what makes a `.prj` file resolve versus return
`None`.

## Range validation on write into fixed-width columns

Confirmed directly: a column value that doesn't fit its target
`PointAttribute`'s dtype is rejected, not silently wrapped:

```python
>>> list(XYZReader("intensity_70000.xyz"))   # intensity column has value 70000
CorruptedFileError: Column 'intensity' has values out of range for its
uint16 representation: expected [0, 65535], got [70000, 70000].
```

This closes a real, confirmed defect: `ASCIIConverter` is a
standalone implementation (like `LASConverter`) that does not go
through `topocore.io.common.base_converter.BasePointConverter`, so
the range-validation fix already applied there did not protect ASCII
sources at all until this was fixed. Before the fix, an intensity
value of `70000` silently became `4464` (`70000 % 65536`) via a
plain `.astype()` cast — no error, no warning. `NaN`/infinite values
destined for an integer column are also rejected explicitly, since
NumPy's own float→int cast silently turns `NaN` into `0` with only a
`RuntimeWarning`, not an exception.

## Errors

| Exception | When |
|---|---|
| `CorruptedFileError` | Range violation (above); PTS count issues (above); a malformed record |
| `InvalidASCIIRecordError` | A record cannot be parsed into the expected column structure |
