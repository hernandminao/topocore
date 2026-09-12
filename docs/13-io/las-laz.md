# LAS / LAZ

```text
LAS/LAZ file
      │
      ▼
LASReader / LAZReader  (streams via laspy's chunk_iterator)
      │
      ▼
LASConverter.from_las_points()
      │
      ▼
   Chunk  (X, Y, Z, INTENSITY, CLASSIFICATION, RETURN_NUMBER,
           NUMBER_OF_RETURNS, GPS_TIME, COLOR — whichever the
           source file actually has)
      │
      ▼
   PointCloud
```

LAZ is compressed LAS — same point record format, same converter
(`LAZReader._iterate_chunks()` imports and reuses
`topocore.io.las.converter.LASConverter` directly). The only
difference is the compression backend (`lazrs`) and its own error
message when that backend is missing or the file can't be decoded.

## Reading

```python
from topocore.io.las import LASReader
# or: from topocore.io.laz import LAZReader

reader = LASReader("survey.las", chunk_size=100_000)
for chunk in reader:
    ...
reader.close()
```

`chunk_size` (default: `topocore.io.constants.DEFAULT_CHUNK_SIZE`)
controls how many points are read per iteration. Both readers reject
`chunk_size <= 0` at construction:

```python
>>> LASReader("x.las", chunk_size=0)
ValueError: chunk_size must be greater than zero.
```

### Attribute mapping

| LAS field | `PointAttribute` |
|---|---|
| `x`, `y`, `z` | `X`, `Y`, `Z` |
| `intensity` | `INTENSITY` |
| `classification` | `CLASSIFICATION` |
| `return_number` | `RETURN_NUMBER` |
| `number_of_returns` | `NUMBER_OF_RETURNS` |
| `gps_time` | `GPS_TIME` |
| `red`, `green`, `blue` (all 3 present) | `COLOR` (shape `(n, 3)`, combined) |

Only attributes actually present in the source file are included in
each `Chunk` — confirmed directly: a LAS file written without color
data produces chunks with no `COLOR` attribute at all, not a
zero-filled one.

### A confirmed current limitation: `.header` is always `None`

`BaseLASReader` exposes a `header` property, but confirmed directly:
for both `LASReader` and `LAZReader`, `.header` returns `None` —
before opening the file, during iteration, and after it completes.
Neither reader's own `_open()` implementation ever assigns
`self._header`, even though `self._reader` (laspy's own file handle,
which does carry a real header) is set correctly. If you need header
information (point counts, bounds, scale/offset as stored in the
file), read it directly via `laspy.open(path).header` rather than
through this property.

## CRS detection

Both readers carry native CRS detection — a file's own embedded VLR
(GeoTIFF-keys or WKT, depending on point format/version) is read via
`laspy`'s own `LasHeader.parse_crs()`. If no native CRS is present,
`topocore.io.crs.detect_crs()` (or `apply_crs_with_native_priority()`
internally) falls back to a `.prj` sidecar. A native CRS, once found,
is never overridden by a `.prj` — even one declaring something
different. See [`crs-detection.md`](./crs-detection.md) for the full
architecture, priority rules, and how a detected CRS ends up on
`PointCloud.crs`.

No CRS present, or a structurally broken CRS VLR, resolves to `None`
— never an error. A compound horizontal+vertical CRS
(`EPSG:32618+5773`) is detected correctly, though its own `.epsg`
comes back `None` (inherent to `pyproj.CRS.to_epsg()` for a compound
CRS, not a limitation TopoCore introduces).

### Errors

```python
>>> LASReader("does_not_exist.las").read()
PointCloudIOError: File not found: does_not_exist.las

>>> LAZReader("corrupted.laz").read()
PointCloudIOError: Unable to open the LAZ file. Ensure the file is
valid and the LAZ backend (lazrs) is correctly installed.
```

## Writing

```python
from topocore.io.las.writer import LASWriter
# or: from topocore.io.laz import LAZWriter

writer = LASWriter("output.las", point_format=3, version="1.2")
writer.write(cloud)
```

**`LASWriter` is not exported from `topocore.io.las`'s public
`__all__`.** The package's own docstring states explicitly: "only
`LASReader` is considered part of the public API." `LASWriter` is
fully implemented and used internally (`LAZWriter` is a structural
near-duplicate of it), and is importable directly
(`from topocore.io.las.writer import LASWriter`) — but it does not
carry the same API-stability guarantee as `LASReader`. `LAZWriter`,
by contrast, *is* `laz`'s own declared public API.

### Coordinate precision: 1mm default

```python
LASWriter(path, *, point_format=3, version="1.2", scale=None, offset=None)
```

If `scale` is not given, both writers default to **1mm** per axis —
the ASPRS-recommended default — not laspy's own coarser 1cm internal
default. This was a real, confirmed defect until PR19: the writer
previously never set `header.scales` at all, silently relying on
laspy's 1cm default for every write, regardless of the source data's
actual precision. A write → read round trip on realistic UTM-scale
coordinates showed the effect concretely: `500123.456` became
`500123.46` — a tenth of a millimeter's worth of coordinate destroyed
silently, in a library explicitly built around millimeter-precision
GNSS RTK survey workflows.

Confirmed directly, with the current (fixed) writer: a round trip on
realistic coordinates now preserves precision to well within 1mm
(observed deviation ~0.0002, consistent with 1mm-scale integer
encoding — not the old ~0.01 error).

If `offset` is not given, it's computed automatically from the
minimum X/Y/Z across the data being written, keeping the internal
scaled-integer values small.

### Writing performance: streamed, not merged

`write()` writes each `Chunk` incrementally via laspy's low-level
`LasWriter.write_points()`, rather than first concatenating every
chunk's arrays into one giant array per attribute. This was a
deliberate PR21.7.3 change: benchmarking showed the old
merge-then-write approach's memory overhead grew worse than linearly
with point count (1.4x at 100,000 points, 2.8x at 5,000,000) relative
to the underlying chunk data — confirmed to produce numerically and
semantically identical output to the old approach via a full
write → read → compare round trip.

### Verified: LAZ compression ratio

A 5,000-point cloud (X/Y/Z/CLASSIFICATION) written both ways:
LAS: 170,227 bytes; LAZ: 36,143 bytes — roughly a 79% size reduction,
confirmed directly, not assumed from LAZ's general reputation for
compression.
