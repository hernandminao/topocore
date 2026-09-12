# `topocore.dxf` — Overview

## Scope

`topocore.dxf` exports a `FeatureCollection` (the output of
`topocore.features`) to a real, `ezdxf`-backed DXF file, readable by
AutoCAD/Civil 3D and other CAD software. `ezdxf` is an optional
dependency (`pip install topocore[dxf]`) -- `DXFExporter.__init__()`
raises `DXFExportError` immediately, with a clear install message, if
it is not present, confirmed by direct execution and mirroring the
identical pattern `topocore.gpkg` does not need (that package uses
only mandatory dependencies).

## Real inventory — 13 files, 1009 lines

```text
topocore/dxf/
├── __init__.py       exports DXFExporter and its supporting models/exceptions
├── _ezdxf_compat.py    is_available() / require_ezdxf() -- the optional-dependency boundary
├── constants.py          DXF version, APPID, schema version
├── entities.py             write_entity() -- FeatureGeometry -> ezdxf entities
├── exceptions.py            DXFError -> DXFExportError/DXFGeometryError/DXFValidationError
├── exporter.py               DXFExporter -- the orchestrator
├── layers.py                  layer/color assignment, contour MAJOR/MINOR resolution
├── mapping.py                  GeometryMapper -- geometry type -> DXF representation
├── models.py                    DXFExportOptions, ExportContext, LayerStyle, etc.
├── report.py                     DXFExportReport / _ReportBuilder
├── tolerance.py                   DXFTolerance -- planarity/coordinate tolerances
├── validation.py                  DXFValidator -- pre-write, per-feature checks
└── xdata.py                        typed Feature-attribute round-trip via DXF XDATA
```

No `PointCloud -> DXF` capability exists, matching `topocore.gpkg`'s
own identical scope boundary -- this package also consumes
`FeatureCollection` exclusively, confirmed by reading every import
in the package.

## Architecture — export flow

```text
FeatureCollection
      |
      v
for each Feature:
   DXFValidator.validate()  -- per-feature, ERROR/WARNING severities
      |
      +-- ERROR issues --> strict=True: raise DXFValidationError (abort)
      |                    strict=False: skip, record, continue
      v
   GeometryMapper.decide()  -- geometry_type -> POINT/LWPOLYLINE/POLYLINE3D/3DFACE
   _resolve_layer()          -- cad_layer attribute, or CONTOUR-specific
      |                          MAJOR/MINOR resolution, or LAYER_BY_FEATURE_TYPE
      +-- DXFGeometryError/DXFExportError --> strict-aware skip-or-abort,
      |    same as above
      v
   write_entity()            -- ezdxf entity + typed XDATA under APPID "TOPOCORE"
      |
      v
write to a temp file (same directory as the target path)
      |
      v
os.replace(temp, final_path)  -- atomic; confirmed no orphaned temp file
                                  on success, no file at all on failure
```

Confirmed directly, matching `gpkg`'s own equivalent structure: the
`strict`-aware skip-or-abort logic lives entirely in `export()`'s own
per-feature loop, and the final write is atomic.

## This package's own recurring historical pattern -- and this audit's own contribution to it

`topocore.dxf`'s own source contains 2 already-documented, already-fixed
historical defects, each with its own explanatory comment written directly
in the code:

- **PR19** (`layers.py`'s own `layer_for()`): a raw `KeyError`, for any
  `FeatureType` absent from `LAYER_BY_FEATURE_TYPE` (63 of 84 types,
  confirmed by the comment's own count), used to crash the entire
  export regardless of `strict`. Fixed by wrapping it into
  `DXFExportError` at the call site in `_resolve_layer()`.
- **PR20** (`workflow.py`'s own `export_dxf()`): `**exporter_kwargs`
  used to be passed directly to `DXFExporter.__init__()`, which does
  not accept arbitrary keyword arguments -- `export_dxf(path,
  strict=False)` crashed immediately. Fixed by building a proper
  `DXFExportOptions` from the kwargs first, mirroring `export_gpkg()`'s
  own already-correct pattern.

**This audit found and fixed a 3rd instance of the exact same
recurring pattern** (an unwrapped, non-domain exception escaping
`DXFExporter.export()`'s own `strict`-aware boundary) -- see
[`validation.md`](./validation.md) for the complete account. Finding
this pattern a 3rd time, in a package whose own source already
documents 2 prior occurrences, is itself informative: it suggests
this specific failure shape ("a helper function raises a raw
exception type that `export()`'s own except-tuple doesn't list") is
this package's own most persistent, recurring risk -- confirmed
worth explicitly checking for on any FUTURE change to this exporter,
not only during an audit like this one.

## Investigated and confirmed correct, not defects

- **`write_entity()`'s own `else: raise NotImplementedError(...)`**
  (a 4th, unhandled `DXFRepresentation` case): confirmed, by direct
  execution, that `GeometryType`'s own 4 real members map exactly,
  one-to-one, onto `GeometryMapper.decide()`'s own 4 possible
  `DXFRepresentation` outputs -- no legitimately-constructed `Feature`
  can reach this branch today. Left as internal defensive code, not
  treated as a defect requiring a fix, per this audit's own explicit
  reachability standard (a bare `raise` is not itself sufficient
  evidence of a defect -- see [`limitations.md`](./limitations.md)
  for why it is still worth knowing about).
- **`XDataEncoder.encode()`'s own `TypeError`** for a non-scalar
  value: confirmed, by direct execution with a `Feature` carrying a
  nested-dict attribute, that `build_feature_xdata()`'s own
  pre-filtering (`is_xdata_encodable()`) already removes anything
  that would trigger it, before `XDataEncoder.encode()` is ever
  called from the real `export()` flow. Same disposition as above.
- **`strict=True`'s own abort behavior**: confirmed unchanged by this
  audit's own fix -- still aborts immediately, still leaves no file,
  for the identical invalid input that `strict=False` now correctly
  isolates.
- **Atomic write**: confirmed directly, both before and after this
  audit's own fix -- a successful export leaves exactly the target
  file; a failed one (an invalid feature under `strict=True`) leaves
  none.

## Where to go next

- [`validation.md`](./validation.md) -- the 1 fixed defect, in full,
  with the same evidence chain used throughout this project.
- [`limitations.md`](./limitations.md) -- pre-existing, out-of-scope
  findings only.
