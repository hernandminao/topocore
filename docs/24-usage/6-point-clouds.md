# Point Clouds

## The reader contract

Every format-specific reader implements the same abstract base,
`topocore.io.base.PointCloudReader`:

```python
class PointCloudReader(ABC):
    def __init__(self, path: str | Path) -> None: ...
    def __iter__(self) -> Iterator[Chunk]: ...   # streams chunks
    def read(self) -> PointCloud: ...             # reads everything at once
    def __enter__(self) -> PointCloudReader: ...
    def __exit__(self, ...) -> None: ...
```

Use as a context manager:

```python
from topocore.io.las import LASReader

with LASReader("survey.las") as reader:
    cloud = reader.read()
```

`Workflow.read_point_cloud()` (see [`workflows.md`](./workflows.md))
selects the reader class by file extension automatically -- use the
readers directly only when you need a specific reader's own
non-default parameters, or want to stream chunks (`for chunk in
reader:`) instead of materializing the whole cloud at once.

## The 7 supported formats — real constructors

Confirmed directly against the source: only `E57Reader`'s
`chunk_size` has no default -- every other reader supplies one.

```python
from topocore.io.las import LASReader
LASReader(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE)

from topocore.io.laz import LAZReader
LAZReader(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE)

from topocore.io.e57 import E57Reader
E57Reader(path: str | Path, *, chunk_size: int)  # required, no default

from topocore.io.ply.reader import PLYReader
PLYReader(path: str | Path, *, chunk_size: int = 1_000_000)

from topocore.io.ascii.xyz import XYZReader
XYZReader(path: str | Path, *, chunk_size: int = 1_000_000, encoding: str = "utf-8")

from topocore.io.ascii.csv.reader import CSVReader
CSVReader(path: str | Path, *, chunk_size: int = 1_000_000, encoding: str = "utf-8", has_header: bool | None = None)

from topocore.io.ascii.pts.reader import PTSReader
PTSReader(path: str | Path, *, chunk_size: int = 1_000_000, encoding: str = "utf-8")
```

`CSVReader`'s own `has_header=None` means auto-detect; pass `True`/
`False` to force a specific interpretation of the first row.

### Writers

2 formats expose a **public** writer, confirmed exported from their
own package:

```python
from topocore.io.laz import LAZWriter
from topocore.io.ascii.xyz import XYZWriter
```

A `LASWriter` class also exists in the source
(`topocore/io/las/writer.py`), but confirmed directly by import, it
is **not** part of the public API -- `from topocore.io.las import
LASWriter` raises `ImportError`; the module's own docstring states
only `LASReader` is public. There is no PLY writer at all, public or
private -- confirmed directly, no `writer.py` exists anywhere under
`topocore/io/ply/`. Neither of the other ASCII formats (CSV, PTS)
exposes a writer either.

## Reading a field survey (not a point cloud)

```python
from topocore.survey.reader import SurveyTXTReader

reader = SurveyTXTReader(
    path: str | Path,
    *,
    format: SurveyFormat | None = None,
    delimiter: str | None = None,
    encoding: str = "utf-8",
)
survey_points = reader.read()  # -> SurveyPointSet
```

`SurveyTXTReader` reads a total-station/GNSS survey text file --
`id, x, y, z, code` in any delimited layout (comma, semicolon, tab,
or whitespace), with or without a header row. Column order is
resolved in this order:

1. An explicit `format` preset (`SurveyFormat`), if given.
2. Auto-detection from a recognized header row.

If neither applies, `read()` raises `SurveyFormatError` rather than
guessing at column meaning.

If a `.prj` sidecar file exists next to the survey file and resolves
to a real CRS, `SurveyPointSet.crs` is set from it automatically --
confirmed this is the *only* way a survey text file can carry CRS
information, since plain `id, x, y, z, code` has no CRS concept of
its own. Coordinates themselves are never touched by this: `.prj`
only annotates what CRS the already-read values are declared to be
in, never transforms them.

Use `.path`/`.format` (read-only properties) to inspect what a
reader was constructed with.

## The `PointCloud` container

```python
from topocore.pointcloud.pointcloud import PointCloud

cloud = PointCloud()          # starts empty
len(cloud)                    # number of chunks (not points)
cloud.point_count              # total point count, across all chunks
cloud.crs                       # str | None -- readable and settable
cloud.crs = "EPSG:4326"
for chunk in cloud:              # iterate chunks directly
    ...
cloud.add_chunk(chunk)
```

Confirmed directly: `PointCloud` maintains an internal version
counter, incremented on every operation that changes which points it
holds (`add_chunk`/`remove_chunk`/`clear`) -- this exists specifically
so external caches keyed on a `PointCloud` object (e.g. normal-vector
caching) can detect "same object, mutated in place," which bare
`id(cloud)` cannot (Python object identity never changes when an
object is mutated rather than replaced). This closed a real,
previously demonstrated bug where a cache returned a stale normal
vector after a caller mutated the same `PointCloud` between calls.

## Point attributes

Each `Chunk` holds one or more of these (`topocore.pointcloud.attributes.PointAttribute`):

```text
X, Y, Z
INTENSITY
RETURN_NUMBER, NUMBER_OF_RETURNS
CLASSIFICATION
SCAN_ANGLE
GPS_TIME
COLOR
NIR
NORMAL
USER_DATA
POINT_SOURCE_ID
```

Each is marked `REQUIRED`, `OPTIONAL`, or `FORMAT_DEPENDENT` --
whether a given reader actually populates a given attribute depends
on what the source file itself provides (e.g. `INTENSITY`/`COLOR`
are commonly present in LAS/LAZ, rarely in a plain XYZ text file).

## Where to go next

- [`workflows.md`](./workflows.md) -- `read_point_cloud()`/
  `read_survey()`'s own format dispatch and full stage contract.
- [`geodesy.md`](./geodesy.md) -- `PointCloud.crs`'s own detection
  and propagation, in full, including automatic CRS detection from
  the file itself.
- [`feature-builder-cases.md`](./feature-builder-cases.md) -- why
  these 7 formats never carry a field code, and what to use instead
  (`SurveyTXTReader`) if your data does.
- [`terrain.md`](./terrain.md) -- what happens after ground
  classification (`TIN`/`DTM`).
- [`features.md`](./features.md) -- `detect_features()`'s own
  `DetectionContext`, built from a `PointCloud`.
