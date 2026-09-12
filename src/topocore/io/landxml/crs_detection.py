"""
topocore.io.landxml.crs_detection
====================================

Attempts to resolve a real `CRS` from a LandXML file's own
`<CoordinateSystem>` element. Same public contract as
`topocore.io.las.crs_detection.detect_crs()`/
`topocore.io.e57.crs_detection.detect_crs()`
(`path -> CRS | None`, absence is never an error) -- built on top of
`LandXMLReader`, which already extracts `<CoordinateSystem>` as a raw
string (see `LandXMLDocument.crs`, documented in
`../../14-geodesy/io-boundary.md` and `../../12-landxml/units-and-crs.md`)
but deliberately never resolves it. This module is that resolution
step, kept separate rather than folded into `LandXMLReader` itself --
matching this project's own established pattern of keeping format
parsing (`io/`) and CRS resolution (`geodesy/`-adjacent concerns)
distinct.

The raw string takes 3 different shapes in practice, confirmed
directly against real LandXML files exercising each of
`<CoordinateSystem>`'s own 3 identifying attributes
(`name`/`epsgCode`/`desc`, in that priority order -- see
`LandXMLReader._read_units_and_crs()`):

- `"EPSG:32617"` (from `name`) -- parsed via the `EPSG:` prefix.
- `"2274"` (from `epsgCode`, when `name` is absent) -- a bare integer,
  parsed as an EPSG code directly.
- `"NAD83 State Plane Texas"` (from `desc`, when neither `name` nor
  `epsgCode` is present) -- free-text, human-readable only. Confirmed
  directly: this is not parseable as an EPSG code or as WKT, and
  `detect_crs()` correctly returns `None` for it rather than
  guessing.

A LandXML file with no `<CoordinateSystem>` element at all also
resolves to `None` -- `LandXMLDocument.crs` is `None` in that case.

Additional findings from this module's own audit
------------------------------------------------------
- **Attribute priority is deterministic, confirmed with genuinely
  conflicting values**: if both `name` and `epsgCode` are present and
  disagree, `name` always wins -- inherited directly from
  `LandXMLReader._read_units_and_crs()`'s own
  `name or epsgCode or desc` short-circuit, `epsgCode` is never even
  consulted once `name` is truthy.
- **Namespace version is not a limitation.** Confirmed directly with
  a `LandXML-1.0` namespace and with no namespace declared at all --
  both resolve identically to a `LandXML-1.2`-namespaced file,
  because element matching throughout `LandXMLReader` compares only
  the namespace-stripped local tag name (`local_tag()`), never the
  namespace URI itself.
- **A `<CoordinateSystem>` that is not a direct child of the
  `<LandXML>` root element is genuinely missed.** Confirmed directly:
  nesting it inside e.g. `<Project>` causes `detect_crs()` to return
  `None` even though the file does declare a real CRS -- inherited
  directly from `children()`'s own direct-children-only search,
  the same scope limitation already documented for `<CgPoints>` in
  `../../12-landxml/reader.md`. This is a real, current limitation of
  the underlying reader, not something this module works around.
- **Encoding**: when the file's own XML declaration correctly states
  its actual encoding (confirmed directly with a genuine
  non-ASCII-content ISO-8859-1 file, correctly declared), parsing
  succeeds correctly. When the declared encoding disagrees with the
  file's actual bytes (confirmed directly: implicit UTF-8 default
  against real ISO-8859-1 bytes), the underlying XML parser raises
  `LandXMLParseError` for the resulting invalid byte sequence -- caught
  here and returned as `None`, never a crash.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from topocore.geodesy.exceptions import CRSError
from topocore.io.landxml.exceptions import LandXMLParseError, LandXMLValidationError

if TYPE_CHECKING:
    from topocore.geodesy import CRS

_EPSG_PATTERN = re.compile(r"(?:EPSG:)?(\d+)", re.IGNORECASE)


def detect_crs(path: str | Path) -> CRS | None:
    """
    Attempt to resolve the CRS declared in a LandXML file's own
    `<CoordinateSystem>` element.

    Returns
    -------
    `CRS` if `<CoordinateSystem>` is present and its value can be
    resolved -- either an `EPSG:NNNN`-prefixed string, a bare integer
    (interpreted as an EPSG code), or (as a last attempt) valid WKT.
    `None` in every other case: no `<CoordinateSystem>` element,
    a value that's free text not matching any of the above (e.g. a
    `desc`-only declaration like `"NAD83 State Plane Texas"`), an
    EPSG-shaped number that isn't a real registered code, or a file
    whose own structure prevents reading it at all. Never raises for
    any of these -- an absent or unresolvable CRS declaration is a
    normal, expected outcome, not an error.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    """
    # `LandXMLReader` (and, transitively through `topocore.geodesy`,
    # `CRS` itself) are imported HERE, lazily, rather than at module
    # level -- confirmed directly to be required, not a style choice:
    # `topocore.io.landxml.models` (imported by `LandXMLReader`)
    # depends on `topocore.survey.models`, whose own package
    # `__init__.py` imports `topocore.io.crs.external` ->
    # `topocore.io.crs.detector`, which itself imports THIS module
    # (`topocore.io.landxml.crs_detection`) at module level to
    # register the LandXML branch of `topocore.io.crs.detect_crs()`'s
    # own extension dispatch. A module-level import here completes
    # the cycle: `io.landxml.crs_detection` -> `io.landxml.reader` ->
    # `io.landxml.models` -> `survey.models` -> `survey` (package) ->
    # `io.crs.external` -> `io.crs` (package) -> `io.crs.detector` ->
    # back to `io.landxml.crs_detection`, still mid-load -> ImportError
    # (`cannot import name 'LandXMLDocument' from partially
    # initialized module`). Confirmed to reproduce with EITHER import
    # order (`import topocore.io.landxml` first, or
    # `import topocore.io.crs` first) in a fresh process -- not an
    # artifact of any one entry point, and not something prior
    # testing in this same project caught, since every earlier manual
    # verification happened to import a different, heavier module
    # (`topocore.workflow.workflow`/`topocore.geodesy`) FIRST, which
    # fully populated `sys.modules` for this whole dependency web
    # before `io.landxml`/`io.crs` were ever touched, masking the
    # cycle. Deferring both imports to here (function-call time,
    # after the whole package tree has finished its own top-level
    # loading) matches this project's own established pattern for
    # breaking exactly this class of cycle elsewhere (e.g.
    # `topocore.io.crs.detector`'s own per-extension dispatch
    # imports).
    from topocore.geodesy import CRS
    from topocore.io.landxml.reader import LandXMLReader

    resolved_path = Path(path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"LandXML file not found: '{resolved_path}'.")

    try:
        document = LandXMLReader(resolved_path).read()
    except (LandXMLParseError, LandXMLValidationError):
        return None

    raw_value = document.crs

    if raw_value is None or not raw_value.strip():
        return None

    epsg_match = _EPSG_PATTERN.fullmatch(raw_value.strip())
    if epsg_match:
        try:
            return CRS.from_epsg(int(epsg_match.group(1)))
        except CRSError:
            return None

    try:
        return CRS.from_wkt(raw_value)
    except CRSError:
        return None


__all__ = ["detect_crs"]
