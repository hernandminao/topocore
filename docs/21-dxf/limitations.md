# Limitations

## 2 structurally-unreachable defensive `raise` statements -- not defects, recorded for future awareness

Both investigated with the same reachability standard as this
audit's own 1 fixed defect (see [`validation.md`](./validation.md)),
and both confirmed NOT currently reachable via any legitimately-
constructed `Feature`:

- `entities.py`'s `write_entity()`: `else: raise
  NotImplementedError(f"No entity writer for {decision.representation}.")`.
  Unreachable today because `GeometryType`'s 4 real members map
  exhaustively, one-to-one, onto `GeometryMapper.decide()`'s own 4
  possible outputs.
- `xdata.py`'s `XDataEncoder.encode()`: `raise TypeError(...)` for a
  non-scalar, non-flat-collection value. Unreachable today via the
  real `export()` flow because `build_feature_xdata()`'s own
  `is_xdata_encodable()` pre-filter already removes anything that
  would trigger it.

**Not corrected during this audit**, and not because they are
unimportant: this project's own explicit standard, established
across `18-features` and reinforced here, is that a bare `raise`
statement is not by itself sufficient evidence of a defect --
reachability from the real public API must be demonstrated first,
exactly as it was for `is_index_contour()`'s own now-fixed defect.
Both remain worth knowing about for a different reason: they are
*structurally* the same shape as this package's own 3 confirmed,
real historical defects (`layer_for()`'s PR19 fix,
`export_dxf()`'s PR20 fix, and `is_index_contour()`'s fix in this
audit) -- an internal helper raising an exception type
`DXFExporter.export()`'s own except-tuple does not list. Should
`GeometryType` or the XDATA-encodable type set ever gain a new case
without `write_entity()`/`XDataEncoder.encode()` being updated in
lockstep, this is exactly the shape the resulting bug would take.
Recorded here as a known structural risk to re-check on any future
change to either `GeometryType` or the XDATA type contract -- not as
present technical debt, since nothing currently reachable exercises
either path incorrectly.

## Confirmed pre-existing, out of this audit's own 1-defect scope

`ruff`, with the real project's own `pyproject.toml`, reports 9
findings across `topocore/dxf/`, in files this audit's own fix never
touched (the fix was confined entirely to `layers.py`):

- **7 `RUF022`/`RUF023`** (`__all__`/`__slots__` not alphabetically
  sorted), across `__init__.py`, `constants.py`, `exceptions.py`,
  `mapping.py` (both its `__slots__` and its own `__all__`),
  `models.py`, and `validation.py`. Confirmed the same deliberate
  semantic/logical grouping already established and preserved
  elsewhere in this project (`analysis`, `terrain`, `features`,
  `gpkg`) -- not flattened.
- **1 `UP035`** (`report.py`: prefer `collections.abc.Mapping` over
  `typing.Mapping`): the same mechanical modernization category
  already documented in `16-analysis` (`UP040`) -- cosmetic, zero
  behavior change, left untouched to keep this audit's own scope to
  the 1 agreed defect.
- **1 `SIM102`** (`validation.py`: a nested `if` that could be
  combined into one `if ... and ...`): confirmed purely stylistic --
  the current form matches its own sibling checks in the same
  function (`DXFValidator.validate()` checks `geometry_type` first,
  then a type-specific condition, consistently across all of its own
  branches), so combining only this one would make it inconsistent
  with the rest of the same function, not clearer.

`mypy`, same real configuration, across all 13 real `dxf` source
files: clean -- 0 findings, including in `layers.py` itself after
this audit's own fix.

## No test suite was available for this package

Confirmed directly, not assumed from an empty search: no
`tests/dxf/` (or equivalent) directory was present in the
reconstructed repository export this audit worked from, and none was
otherwise provided -- the same situation already noted for
`18-features`/`19-gpkg`. The 13 regression tests this audit added
(see [`validation.md`](./validation.md)) are, as far as this audit
could confirm, the first tests this package has had.

## What this audit did not attempt

A byte-level comparison of this project's own generated `.dxf` files
against an AutoCAD/Civil-3D-round-tripped reference file was not
performed -- this audit confirmed correctness by direct execution
(entity counts, layer assignment, XDATA round-trip via
`XDataEncoder`/`XDataDecoder`, atomic write) and by reading the
entity-writing code directly against `ezdxf`'s own documented API
usage, not by opening the output in real CAD software. No evidence
of an incompatibility was found during this audit's own direct
inspection, but this is a narrower form of verification than an
actual CAD round-trip would provide -- the same caveat already
recorded for `19-gpkg`'s own GDAL/OGR comparison.
