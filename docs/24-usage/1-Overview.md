# TopoCore — Overview

Every signature in this guide was extracted directly from the real
source (confirmed during PR22's own audit) -- not written from
memory. Where a signature has changed or a defect was fixed during
that audit, this guide reflects the corrected, current behavior.

## Conceptual pipeline

```text
Input file (LAS/LAZ/E57/PLY/XYZ/CSV/PTS, or a survey text file)
      |
      v
PointCloud  /  SurveyPointSet
      |
      +-- CRS / Geodesy (transform_crs, transform_vertical, georeference)
      +-- Classification (ground, multi-class)
      +-- Normals / PCA / curvature (topocore.processing.features)
      +-- Sampling, segmentation, registration
      |
      v
Terrain                              Features
  TIN -> DTM -> Contours               detect_features() (from PointCloud)
                                        build_features_from_survey() (from
                                        SurveyPointSet, via field codes)
      |                                     |
      +---------------------+--------------+
                             v
                      FeatureCollection
                             |
      +----------------------+----------------------+
      v                                              v
  Analysis                                        Export
    distance, profile, visibility,                  DXF
    statistics, quality                             GeoPackage
```

## Two ways to work

**High-level orchestrator** -- `Workflow`, a fluent, chainable API that
manages artifacts, dependencies, versions, staleness detection,
execution history, and progress reporting for you:

```python
from topocore.workflow import Workflow

workflow = Workflow()
workflow.read_point_cloud("survey.las")
workflow.classify_ground()
workflow.build_tin()
```

**Modular API** -- every module `Workflow` delegates to is usable
directly, with no orchestration overhead:

```python
from topocore.io.las import LASReader
from topocore.processing.ground import GroundManager
from topocore.terrain.tin import TIN

with LASReader("survey.las") as reader:
    cloud = reader.read()

ground_cloud = GroundManager().extract(cloud)
tin = TIN.from_points(tuple(...))  # from your own point construction
```

`Workflow` is not a separate implementation of these algorithms --
confirmed directly during this project's own audit: every stage
method's own `work()` closure calls exactly the same modular classes
shown above. Choosing `Workflow` buys you dependency tracking,
staleness detection, and a uniform error/history contract; choosing
the modular API buys you direct control with no wrapping at all.

## Where to go next

- [`quickstart.md`](./quickstart.md) -- a complete, minimal, real
  end-to-end pipeline using `Workflow`.
- [`end-to-end-examples.md`](./end-to-end-examples.md) -- every
  input format, loader, CRS step, processing stage, analysis call,
  and export configuration, each verified with real execution.
- [`reference.md`](./reference.md) -- a one-page lookup table from
  task to real API call.
- [`workflows.md`](./workflows.md) -- every `Workflow` stage method,
  in full, including error/recovery contracts confirmed during audit.
- [`point-clouds.md`](./point-clouds.md) -- the 7 supported point-cloud
  formats and `SurveyTXTReader`.
- [`primitives.md`](./primitives.md) -- `Point3D`, `BBox3D`, `Vector3D`,
  and building a `PointCloud`/`TIN` from raw data, no file required.
- [`landxml.md`](./landxml.md) -- reading/writing LandXML surfaces,
  point groups, and alignments.
- [`alignment.md`](./alignment.md) -- the full horizontal/vertical
  alignment domain (`ArcElement`, `SpiralElement`, `DesignProfile`),
  independent of LandXML.
- [`geodesy.md`](./geodesy.md) -- `CRS`, coordinate/vertical
  transforms, UTM, automatic CRS detection, and control-point
  georeferencing.
- [`features.md`](./features.md) -- `Feature`/`FeatureCollection`,
  detectors, and the feature-code catalog system.
- [`feature-builder-cases.md`](./feature-builder-cases.md) -- every
  real `FeatureBuilder` case (LINE/POINT/POLYGON/GROUND/unregistered),
  and whether CSV/PTS/XYZ apply to field-coded surveys (they don't).
- [`terrain.md`](./terrain.md) -- `TIN`/`DTM`/contours.
- [`analysis.md`](./analysis.md) -- distance, profile, visibility,
  statistics, quality.
- [`surface.md`](./surface.md) -- surface comparison and cut/fill.
- [`export.md`](./export.md) -- `DXFExporter`/`GeoPackageExporter`,
  including the `strict` contract confirmed during audit.
- [`catalogs.md`](./catalogs.md) -- `FeatureCodeRegistry` and external
  catalog loading.
