# CRS Detection vs. CRS Resolution: Two Different Boundaries

An earlier version of this page described "no reader ever resolves a
CRS" as an absolute architectural fact. That is no longer accurate —
CRS detection now exists across every point-cloud format and
LandXML. This page documents the boundary as it actually stands
today: what each format resolves automatically, and what genuinely
still requires explicit action from the caller.

## What is now automatic

Every reader `topocore.io` provides — LAS, LAZ, E57, PLY, XYZ, CSV,
PTS, Survey TXT, and LandXML (via a dedicated resolver) — can
populate a real CRS on the artifact it produces, either from the
format's own native mechanism (LAS/LAZ, E57, LandXML) or from a
`.prj` sidecar file (every format that has no native mechanism of
its own). The complete architecture, per-format priority, and exact
conditions under which detection returns `None` instead of a `CRS`
are documented in
[`../13-io/crs-detection.md`](../13-io/crs-detection.md) — this page
does not repeat that detail.

```python
from topocore.workflow import Workflow
from topocore.workflow.artifacts import ArtifactType

workflow = Workflow().read_point_cloud("survey.xyz")  # survey.prj sits alongside it
cloud = workflow.artifact(ArtifactType.POINT_CLOUD)
cloud.crs   # 'EPSG:32617' -- resolved automatically, no separate call needed
```

Detection is not universal, though — it depends on what the source
file itself actually declares:

- A file with no native CRS and no `.prj` sidecar produces
  `.crs is None` — TopoCore never guesses or defaults to one.
- E57's own native mechanism (`coordinateMetadata`) is genuinely
  unreliable in practice — most real-world E57 files never had a CRS
  written into them at all. See
  [`../13-io/e57.md`](../13-io/e57.md).

## What detection does *not* do

**Detecting a CRS is not the same as transforming coordinates, and
it is not the same as georeferencing.** These remain 3 separate
operations (see [`overview.md`](./overview.md) for all 4, including
declaration/propagation):

```text
Detection                Transformation              Georeferencing
"what CRS does           "convert from a             "estimate a
 this file already        known source CRS            transformation from
 declare?"                 to a known target           local coordinates,
                            CRS"                        using control points"
detect_crs(path)         CoordinateTransformer      fit_georeferencing()
-> CRS | None            Workflow.transform_crs()   Workflow.georeference()
```

If you read a file whose CRS was detected as `EPSG:32618` but you
need it in `EPSG:4326`, you still build the `CRS` objects and a
`CoordinateTransformer` yourself, and call `transform_crs()`
explicitly — detection tells you what CRS you have, it never decides
what CRS you want.

## LandXML's own distinction, preserved

`LandXMLReader.read()` itself still only preserves
`<CoordinateSystem>` as an **opaque, unresolved string** on
`LandXMLDocument.crs` — this has not changed. What's new is a
**separate** function,
`topocore.io.landxml.crs_detection.detect_crs()` (also reachable via
the shared `topocore.io.crs.detect_crs()` facade), which resolves
that same value into a real `CRS`. See
[`../12-landxml/units-and-crs.md`](../12-landxml/units-and-crs.md)
for the full distinction between the two.

## What remains genuinely absent

- **A `Workflow`-level `input_crs`/`processing_crs`/`output_crs`
  orchestration model.** Detection populates an artifact's own
  `.crs` field; `Workflow` itself still has no concept of "the
  pipeline's current CRS" beyond whatever each artifact's own field
  says. A genuine architectural decision, not attempted here.
- **A `CRSStatus.CONFLICT` mechanism for reconciling disagreeing CRS
  sources.** There is no case today where a native CRS and a `.prj`
  could compete on the same file (a native CRS, once found, is never
  even compared against a `.prj`), so no conflict-resolution
  mechanism has been built or exercised.
- **Vertical (datum-level) CRS detection.** Detection resolves a
  horizontal CRS; it says nothing about which vertical datum a
  file's own heights are expressed in. See
  [`vertical-reference.md`](./vertical-reference.md).
