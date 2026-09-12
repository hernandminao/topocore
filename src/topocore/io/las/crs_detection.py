"""
topocore.io.las.crs_detection
================================

Detects an embedded CRS from a LAS/LAZ file's own header (WKT or
GeoTIFF VLR keys), via `laspy`'s own `LasHeader.parse_crs()`.

Scope, deliberately narrow
----------------------------
This is the first, smallest slice of a much larger georeferencing
vision discussed for TopoCore (automatic CRS detection across every
IO format, an Input/Processing/Output CRS model on `Workflow`,
`CRSStatus.CONFLICT` resolution across multiple sources, sidecar
files for ASCII formats). None of that broader system exists yet or
is implied by this module. This module does exactly one thing:
attempt to read whatever CRS a LAS/LAZ file's own header already
declares -- nothing more.

**No CRS present is not an error.** Confirmed directly (built a real
LAS file via `laspy` with `header.add_crs()`, and a real one without):
`detect_crs()` returns `None` when nothing usable is present, matching
`parse_crs()`'s own convention and this capability's own explicit
requirement that TopoCore "never invent a CRS" and never treat an
absent one as a failure to process the file.

**A structurally broken file is treated the same as "no CRS", not
as a detection failure.** Confirmed directly: opening a file with an
invalid LAS signature raises `laspy.errors.LaspyException` (a
`LaspyException` subclass) from `laspy.open()` itself, before
`parse_crs()` is ever reached. `detect_crs()` catches
`LaspyException` broadly and returns `None` -- the file's own
structural validity is a separate concern for the actual
point-cloud reader (`LASReader`) to raise on when the caller tries
to read points, not something this narrowly-scoped detector should
surface. A missing file (`FileNotFoundError`) is NOT caught here,
since that's a caller error distinct from "this file has no usable
CRS".

Verified directly for both LAS and LAZ (compression is orthogonal to
the VLR/header structure this relies on), for both the GeoTIFF-keys
VLR path (LAS < 1.4 / point format < 6) and the WKT VLR path (LAS
>= 1.4 / point format >= 6, which `laspy.LasHeader.add_crs()` itself
switches to automatically) -- including a genuinely non-EPSG,
custom-parameter CRS (only representable via the WKT path; `laspy`'s
own GeoTIFF-keys writer explicitly refuses a projected CRS with no
EPSG code) and a compound horizontal+vertical CRS (e.g.
`EPSG:32618+5773`; `epsg` is correctly `None` for the returned `CRS`
in this case -- confirmed this is inherent to `pyproj.CRS.to_epsg()`
itself for a compound CRS, not a limitation introduced here).

Also verified directly: a structurally valid LAS file whose CRS VLR
data itself is corrupted (not merely a file with an invalid overall
signature) -- `parse_crs()` handles this without leaking an
unexpected exception, and `detect_crs()` returns `None`.

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


def detect_crs(path: str | Path) -> CRS | None:
    """
    Attempt to detect the embedded CRS in a LAS/LAZ file's own
    header.

    Returns
    -------
    `CRS` if the file's header declares a CRS `laspy` can parse
    (WKT or GeoTIFF VLR keys), `None` if no CRS is present or the
    file's own structure prevents even checking (e.g. an invalid
    file signature). Never raises for either of these -- "no CRS
    found" is a normal, expected outcome, not an error.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist. This is a genuine caller error,
        distinct from "this file has no usable CRS", and is not
        swallowed.
    """
    import laspy  # type: ignore[import-untyped]
    from laspy.errors import LaspyException  # type: ignore[import-untyped]

    resolved_path = Path(path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"LAS/LAZ file not found: '{resolved_path}'.")

    try:
        with laspy.open(str(resolved_path)) as reader:
            pyproj_crs = reader.header.parse_crs()
    except LaspyException:
        return None

    if pyproj_crs is None:
        return None

    return CRS.from_wkt(pyproj_crs.to_wkt())


__all__ = ["detect_crs"]
