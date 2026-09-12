"""
topocore.io.e57.crs_detection
================================

Attempts to recover a CRS declared in an E57 file's own
`/coordinateMetadata` field. Same public contract as
`topocore.io.las.crs_detection.detect_crs()` (`path -> CRS | None`,
absence is never an error) -- but the reliability underneath is
genuinely different, and this module's own docstring is explicit
about that difference rather than hiding it behind an identical
interface.

Audit findings that shaped this module's own honest scope
-------------------------------------------------------------
Confirmed directly, from 4 independent sources, before writing this
module: unlike LAS's standardized, structured GeoTIFF VLR mechanism,
E57's own `coordinateMetadata` is a **free-text string field** with
no enforced structure -- the ASTM E57 spec itself only says it's
*meant* to hold OGC/WKT-style text, without mandating it. Empirically:

- The official spec paper (Huber, ASTM E57 Committee) describes it as
  an "optional coordinate metadata string", not a structured field.
- MATLAB's own official `readCRS()` documentation example, run
  against MathWorks' own sample E57 dataset, reports
  "No CRS data available."
- A real-world E57 writer (the `PCD-E57` / CloudCompare converter,
  actual C++ source) writes a literal empty string by default, with
  its own code comment reading "Really should be a valid WKT string
  identifying the CRS."
- Confirmed directly with `pye57` itself: even its own default write
  path (`write_scan_raw()`) leaves `coordinateMetadata` as an empty
  string, and its high-level `E57` convenience class locks this field
  to that empty value the moment the file is opened for writing --
  populating it with real WKT requires dropping to `pye57`'s
  low-level `ImageFile` API entirely, which most real-world writers
  will not do.

**This means: for E57, returning `None` will be the overwhelmingly
common outcome in practice** -- not because detection is broken, but
because most real-world E57 files simply never had a CRS written
into them in the first place. This is a materially different
situation from LAS, where a missing CRS more often genuinely reflects
"the surveyor's own software didn't embed one" rather than "the
format itself makes this awkward to populate". Do not present E57
CRS detection to a user as comparably reliable to LAS's.

`coordinateMetadata` is a file-level field, never per-scan
-------------------------------------------------------------
Confirmed directly, including with a genuine 3-scan E57 file:
`/coordinateMetadata` exists exactly once, at the file's own root,
and applies to every `Data3D` scan the file contains uniformly --
no individual scan carries its own separate CRS declaration (each
scan's own `pose` describes its position/orientation relative to a
shared reference frame, not a coordinate system of its own). A
multi-scan E57 file therefore cannot have "conflicting" per-scan CRS
declarations -- this isn't a gap in `detect_crs()`'s own coverage,
it reflects a genuine structural fact about the format itself.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path

from topocore.geodesy import CRS
from topocore.geodesy.exceptions import CRSError


def detect_crs(path: str | Path) -> CRS | None:
    """
    Attempt to recover the CRS declared in an E57 file's own
    `/coordinateMetadata` field, only when it holds text `CRS.from_wkt()`
    can actually parse.

    Returns
    -------
    `CRS` if `coordinateMetadata` is present and holds valid,
    parseable WKT. `None` in every other case: the field is empty
    (confirmed the common case for real-world E57 files -- see this
    module's own docstring), missing entirely, holds text that isn't
    valid WKT, or the file's own structure prevents even checking.
    Never raises for any of these -- an absent or unusable CRS
    declaration is a normal, expected outcome for this format, not
    an error.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist. A genuine caller error, distinct
        from "this file has no usable CRS", and is not swallowed --
        checked explicitly before opening, since `pye57` itself
        raises the same generic exception type for a missing file as
        it does for a structurally invalid one.
    """
    import pye57  # type: ignore[import-untyped]
    from pye57.libe57 import E57Exception  # type: ignore[import-untyped]

    resolved_path = Path(path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"E57 file not found: '{resolved_path}'.")

    try:
        e57_file = pye57.E57(str(resolved_path), mode="r")
        try:
            if not e57_file.root.isDefined("coordinateMetadata"):
                return None
            raw_value = e57_file.root["coordinateMetadata"].value()
        finally:
            e57_file.close()
    except E57Exception:
        return None

    if not raw_value or not raw_value.strip():
        return None

    try:
        return CRS.from_wkt(raw_value)
    except CRSError:
        return None


__all__ = ["detect_crs"]
