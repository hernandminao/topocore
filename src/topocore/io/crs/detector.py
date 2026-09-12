"""
topocore.io.crs.detector
===========================

`detect_crs(path)`: the single, format-agnostic entry point for CRS
detection across every format TopoCore reads. Orchestrates the
already-implemented, already-tested per-format detectors -- this
module re-implements none of their own logic, it only decides which
one to call.

Deliberately out of scope -- see this project's own geodesy
documentation for the full rationale
-------------------------------------------------------------------
- **Transformation.** This detects; it never transforms. Use
  `topocore.geodesy.CoordinateTransformer` (and
  `Workflow.transform_crs()`/`transform_vertical()`) separately, once
  you have a source and target CRS.
- **Control-point georeferencing.** Converting local/arbitrary survey
  coordinates to a known CRS via known control-point correspondences
  (a least-squares similarity/Helmert fit, residual/RMS validation,
  distinguishing control points from independent check points) is a
  third, distinct capability -- not detection, not a plain coordinate
  transform. It does not exist in TopoCore yet. Confirmed during this
  facade's own audit: neither `topocore.analysis.quality.gps_control`
  (which only *evaluates* an already-aligned comparison, producing
  RMSE/pass-fail -- it computes no transformation) nor
  `topocore.processing.registration` (ICP-family point-cloud-to-
  point-cloud alignment, a different problem: unlabeled geometric
  correspondence, not labeled control points with known target
  coordinates) already provide this. When it's built, it can reuse
  `topocore.geodesy.HelmertParameters`/`CoordinateOperation`/
  `CoordinateTransformer.from_operation()` directly (confirmed these
  accept directly-supplied, non-EPSG-registered parameter values --
  a future least-squares fit computing its own `tx`/`ty`/.../`scale`
  is not blocked by anything this facade introduces), and can follow
  `GPSControlResult`'s own existing shape
  (`max_deviation`/`mean_deviation`/pass-fail-per-tolerance) for its
  own residual reporting rather than inventing a new one.

Per-format strategy, confirmed during each format's own audit
-------------------------------------------------------------------
- `.las`/`.laz`: native (embedded VLR/GeoTIFF-key CRS) takes
  priority; falls back to a `.prj` sidecar when absent.
- `.e57`: native (`coordinateMetadata`) takes priority; falls back to
  `.prj` when absent -- including when native detection's own
  confirmed 4-way ambiguity (empty/absent/corrupt/invalid all
  collapsing to `None`) applies; all 4 are treated identically here.
- `.xml`: native (LandXML's own `<CoordinateSystem>`) only, no `.prj`
  fallback -- deferred, since `LandXMLReader` returns a
  `LandXMLDocument`, not a `PointCloud`/`SurveyPointSet`, an
  incompatible shape for today's `.prj`-application machinery.
  **`.xml` does not exclusively identify a LandXML file** -- it's a
  generic container extension shared by unrelated XML documents.
  Confirmed this is safe in practice (LandXML's own detector already
  returns `None` cleanly for a non-`<LandXML>` root element,
  verified directly), but the dispatch itself is an approximation,
  not a guarantee.
- `.ply`/`.xyz`/`.csv`/`.pts`: `.prj` only -- confirmed none of these
  4 formats has any internal CRS mechanism of its own.
- Any other extension: `None`, treated identically to "no CRS
  determinable" -- deliberately not an error. This is a softer
  standard than `Workflow.read_point_cloud()`'s own extension
  dispatch (which does reject unsupported formats outright), since
  this facade's failure mode is informational, never a hard block on
  further processing.

`SurveyTXTReader`'s own `.txt` format is intentionally NOT included
in this dispatch table: its file extension is not reserved or
distinctive (arbitrary `.txt` files are extremely common and mostly
unrelated to survey data), so guessing "this .txt is a survey file"
purely from extension would be exactly the kind of invented signal
this project's whole CRS-detection effort has avoided. Call
`topocore.survey.reader.SurveyTXTReader(path).read().crs` directly
when you know a path is a survey text file.

Performance: header-only, confirmed with a real spy, not assumed
-------------------------------------------------------------------
This never reads point/scan data, regardless of file size. Confirmed
directly with a monkeypatched spy on `laspy.LasReader.chunk_iterator`
(the method that actually reads LAS/LAZ point data): it is never
invoked by this function. The main data file is opened exactly once
(also confirmed directly, via a spy on `laspy.open()`) -- the two
`Path.is_file()` checks (one here, one repeated inside the
format-specific detector this dispatches to) are cheap `stat()` calls,
not file opens, and the `.prj` fallback path (when it applies) opens
only the separate, typically tiny sidecar file -- it never re-opens
the main data file a second time. The same header-only pattern holds
for E57 (`pye57.E57`'s own `ImageFile` only parses the XML section
structure eagerly; binary scan data is read on demand by
`read_scan()`/`read_scan_raw()`, neither of which this ever calls) and
LandXML (its own reader parses the file's XML tree, but never touches
`<CgPoints>`/`<PntList3D>` point *data* extraction paths for a
CRS-only lookup). None of this was benchmarked against a literal
10GB file specifically -- header-only cost does not meaningfully
scale with point-data size, so a dedicated large-file benchmark
would not exercise a different code path than the small fixtures
already used throughout this project's own test suites.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from pathlib import Path

from topocore.geodesy.crs import CRS
from topocore.io.crs.external import ExternalCRSDetector
from topocore.io.e57.crs_detection import detect_crs as _detect_crs_e57
from topocore.io.landxml.crs_detection import detect_crs as _detect_crs_landxml
from topocore.io.las.crs_detection import detect_crs as _detect_crs_las

_NATIVE_WITH_PRJ_FALLBACK = {".las", ".laz", ".e57"}
_EXTERNAL_ONLY = {".ply", ".xyz", ".csv", ".pts"}


def detect_crs(path: str | Path) -> CRS | None:
    """
    Detect the CRS that applies to `path`, dispatching to whichever
    per-format strategy is appropriate for its extension (see this
    module's own docstring for the full table).

    Returns
    -------
    `CRS` if one can be determined by any applicable strategy,
    `None` otherwise -- including for an unrecognized extension.
    Never raises for "no CRS" or "unrecognized format"; both are
    normal, expected outcomes, not errors.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist. Checked unconditionally, before
        any extension-based dispatch, so a missing file is never
        confused with "an unrecognized extension" -- both would
        otherwise resemble each other if this were checked per-branch
        instead.
    """
    resolved_path = Path(path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"File not found: '{resolved_path}'.")

    suffix = resolved_path.suffix.lower()

    if suffix in _NATIVE_WITH_PRJ_FALLBACK:
        native_detector = _detect_crs_las if suffix in (".las", ".laz") else _detect_crs_e57
        native_crs = native_detector(resolved_path)
        return native_crs if native_crs is not None else ExternalCRSDetector().detect(resolved_path)

    if suffix == ".xml":
        # LandXML's own <CoordinateSystem> only -- no `.prj` fallback.
        # Confirmed during this facade's own audit: LandXMLReader.read()
        # returns LandXMLDocument, not PointCloud/SurveyPointSet -- an
        # incompatible shape for today's `.prj`-application machinery
        # (apply_external_crs()/apply_crs_with_native_priority() both
        # require a `.crs` setter that type doesn't have). Adding a
        # fallback here would require a genuinely new mechanism, not a
        # one-line addition -- deliberately deferred, not an oversight.
        return _detect_crs_landxml(resolved_path)

    if suffix in _EXTERNAL_ONLY:
        return ExternalCRSDetector().detect(resolved_path)

    return None


__all__ = ["detect_crs"]
