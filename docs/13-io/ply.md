# PLY

```text
PLY file
    │
    ▼
PLYHeaderParser.parse()   (see coverage-hardening's own audit of this parser)
    │
    ▼
PLYReader  (_iter_ascii / _iter_binary, lazy chunked)
    │
    ▼
PLYConverter.convert()
    │
    ▼
   Chunk  (X, Y, Z, COLOR, NORMAL, plus any generic scalar
           attribute the common mapping recognizes)
    │
    ▼
PointCloud
```

**Read only.** No `writer.py` exists in this package. No optional
dependency — pure standard library (`struct`-level binary reads via
NumPy's `np.fromfile`).

## Supported PLY formats

All 3 storage formats defined by the PLY specification:

| `format` | Supported |
|---|---|
| `ascii` | ✅ |
| `binary_little_endian` | ✅ |
| `binary_big_endian` | ✅ |

## Only the `vertex` element is read

**PLY files with mesh connectivity (an `element face` with a
`property list ... vertex_indices`) have that face data silently
ignored.** `PLYReader` produces a `PointCloud` — points only, never
mesh topology. Confirmed directly: a file with 3 vertices and 1
triangular face reads exactly 3 points; the face's connectivity is
never extracted or represented anywhere in the resulting
`PointCloud`.

If you need mesh faces from a PLY file, this reader does not provide
them — you would need to parse `element face` yourself, or build the
topology separately (e.g. via `terrain.TIN.from_points()`).

```python
>>> reader = PLYReader("mesh_with_faces.ply")
>>> sum(chunk.size for chunk in reader)
3   # vertex count, even though the file also has faces
```

## Attribute mapping

| PLY vertex property | `PointAttribute` |
|---|---|
| `x`, `y`, `z` | `X`, `Y`, `Z` (required — see below) |
| `red`, `green`, `blue` (all 3 present) | `COLOR` (combined) |
| `nx`, `ny`, `nz` (all 3 present) | `NORMAL` (combined) |
| `intensity`, `classification`, `return_number`, `number_of_returns`, `gps_time`/`gpstime`, `user_data`, `point_source_id` | mapped via the shared cross-format attribute table (`topocore.io.common.attribute_mapping`) |

`x`, `y`, and `z` are the only properties genuinely required — every
other property, including color and normals, is optional and simply
absent from the resulting `Chunk` if the file doesn't declare it.

## Reading

```python
from topocore.io.ply.reader import PLYReader

reader = PLYReader("scan.ply", chunk_size=1_000_000)
for chunk in reader:
    ...
```

`chunk_size` (default `1,000,000`) rejects non-positive values at
construction, same as every other reader in this section.

## CRS detection

PLY has no internal CRS mechanism of its own — a `.prj` sidecar
(same base filename, `.prj` extension) is the only source
`detect_crs()` can ever resolve a CRS from for this format. See
[`crs-detection.md`](./crs-detection.md) for the full detail.

## Errors

```python
>>> list(PLYReader("no_xyz.ply"))
MissingAttributeError: PLY vertex element is missing required coordinate properties: x, y, z.

>>> list(PLYReader("no_vertex_element.ply"))
InvalidPLYError: PLY file contains no vertex element.

>>> list(PLYReader("truncated.ply"))
CorruptedFileError: Incomplete PLY ASCII vertex data: header declares 3 vertices, read 2.
```

Header-level structural errors (malformed `format`/`element`/
`property` declarations, unsupported scalar or list types, unknown
keywords) are raised by `PLYHeaderParser` before the reader ever
starts iterating vertex data — see the header parser's own
coverage-hardening audit for the full, exhaustive list of these
(24 confirmed error paths).
