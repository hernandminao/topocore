# Shared Contract: Readers, Writers, Errors

Every format documented in this section — LAS/LAZ, E57, PLY, and the
3 ASCII variants — implements the same two abstract base classes,
`topocore.io.base.PointCloudReader` / `PointCloudWriter`.

## `PointCloudReader`

```python
class PointCloudReader(ABC):
    def __init__(self, path: str | Path) -> None: ...
    @property
    def path(self) -> Path: ...
    def __iter__(self) -> Iterator[Chunk]: ...   # implemented by each format
    def read(self) -> PointCloud: ...            # provided here, calls __iter__
    def close(self) -> None: ...                 # implemented by each format
    def __enter__(self) -> PointCloudReader: ...
    def __exit__(self, ...) -> None: ...          # calls close()
```

`read()` is the one method every reader gets for free from the base
class — it drains `__iter__()` into a single in-memory `PointCloud`.
Prefer iterating directly (`for chunk in reader: ...`) for large
files; `read()` loads everything at once.

Every reader supports the `with` statement, confirmed directly —
`close()` is called automatically on exit:

```python
with XYZReader("cloud.xyz") as reader:
    cloud = reader.read()
# reader is closed here, even if an exception occurred inside the block
```

## `PointCloudWriter`

```python
class PointCloudWriter(ABC):
    def __init__(self, path: str | Path) -> None: ...
    @property
    def path(self) -> Path: ...
    def write(self, cloud: PointCloud) -> None: ...   # implemented by each format
    def close(self) -> None: ...                       # implemented by each format
    def __enter__(self) -> PointCloudWriter: ...
    def __exit__(self, ...) -> None: ...                # calls close()
```

Same `with`-statement support, confirmed directly.

## Consolidated attribute mapping

Different formats recognize different subsets of `PointAttribute`,
using different source-side names. This table consolidates every
format documented in this section (LandXML is excluded — it has no
`PointAttribute` concept at all, see [`../12-landxml/`](../12-landxml/)):

| `PointAttribute` | LAS/LAZ | E57 | PLY | ASCII |
|---|---|---|---|---|
| `X`, `Y`, `Z` | `x`,`y`,`z` | `cartesianX/Y/Z` | `x`,`y`,`z` | `x`,`y`,`z` |
| `INTENSITY` | `intensity` | `intensity` (rescaled) | `intensity` | `intensity` |
| `CLASSIFICATION` | `classification` | — | `classification` | `classification` |
| `RETURN_NUMBER` | `return_number` | — | `return_number` | `return_number` |
| `NUMBER_OF_RETURNS` | `number_of_returns` | — | `number_of_returns` | `number_of_returns` |
| `GPS_TIME` | `gps_time` | — | `gps_time`/`gpstime` | `gps_time` |
| `SCAN_ANGLE` | — | — | — | `scan_angle` |
| `NIR` | — | — | — | `nir` |
| `USER_DATA` | — | — | `user_data` | `user_data` |
| `POINT_SOURCE_ID` | — | — | `point_source_id` | `point_source_id` |
| `COLOR` | `red`+`green`+`blue` | `colorRed`+`colorGreen`+`colorBlue` | `red`+`green`+`blue` | `red`+`green`+`blue` |
| `NORMAL` | — | — | `nx`+`ny`+`nz` | `nx`+`ny`+`nz` |

A `—` means that format's reader has no mapping for that attribute
at all — not that it's merely uncommon in real files. If you need,
for example, per-point normals from a LAS file, this IO layer will
not extract them (LAS's own point record formats generally don't
carry normals anyway).

Every mapping shown above only populates a `Chunk`'s attribute when
the source column/field is actually present — none of these are
zero-filled defaults.

## Error hierarchy

```text
TopoCoreError
    └── PointCloudIOError           -- base for all IO errors, used directly for "file not found"/"couldn't open"
          ├── FileFormatError        -- declared, currently unused (no raises, no subclasses)
          ├── UnsupportedFormatError -- declared, currently unused (no raises, no subclasses)
          ├── CorruptedFileError     -- actively used: truncated/incomplete data, range violations, count mismatches
          ├── InvalidHeaderError     -- never raised directly by its own name, but genuinely exercised
          │     └── PLYError
          │           └── InvalidPLYError  -- PLY's own header-parsing errors (24 confirmed paths)
          ├── MissingAttributeError  -- actively used: PLY's own missing-x/y/z check
          ├── CoordinateSystemError  -- declared, currently unused (no raises, no subclasses)
          └── WriteError             -- declared, currently unused (no raises, no subclasses)

TopoCoreError
    └── ASCIIError                   -- a SEPARATE hierarchy, NOT under PointCloudIOError
          └── InvalidASCIIRecordError
```

**4 of these 8 `PointCloudIOError` subclasses are declared but
currently unused** anywhere in `topocore.io` — confirmed by checking
both direct `raise ClassName(...)` calls and subclass relationships,
not just a literal-name search (which would have missed
`InvalidHeaderError`'s own real usage through `InvalidPLYError`).
This isn't a defect: they read as prepared vocabulary for
capabilities not yet exercised by the current readers/writers (e.g.
no current writer path raises `WriteError` specifically; disk
failures during a LAS/LAZ/ASCII write currently propagate as
whatever the underlying `open()`/`laspy` call itself raises).
**`CoordinateSystemError` remains one of these 4 unused subclasses,
confirmed again after this project's own CRS-detection work
([`crs-detection.md`](./crs-detection.md)):** every detection path —
native (LAS/E57/LandXML) or `.prj` (`ExternalCRSDetector`) — treats
an absent, malformed, or unresolvable CRS declaration as a normal
`None` result, never an exception. `CoordinateSystemError` was
declared for a scenario (a CRS that's actively invalid or
conflicting, rather than merely undetermined) that none of the
current detection code paths raise for. If you're writing code that
should be robust to future format additions, catching the base
`PointCloudIOError` is safer than relying on today's specific subset
being exhaustive.

## A real architectural asymmetry worth knowing

**`topocore.io.ascii.exceptions.InvalidASCIIRecordError` is *not*
part of the `PointCloudIOError` hierarchy at all.** Confirmed
directly via its MRO: it descends from `ASCIIError` straight to
`TopoCoreError`, bypassing `PointCloudIOError` entirely — unlike
`InvalidPLYError`, which genuinely does chain through
`InvalidHeaderError` → `PointCloudIOError`. A caller that catches
only `except PointCloudIOError` around ASCII parsing will **not**
catch `InvalidASCIIRecordError` — only the broader
`except TopoCoreError` (or the ASCII-specific `except ASCIIError`)
would.
