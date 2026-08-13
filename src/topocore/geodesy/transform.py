"""
topocore.geodesy.transform
==============================

Applies an already-built `CoordinateTransformer` to the three
TopoCore models that carry coordinates: `SurveyPointSet`,
`FeatureCollection`, `PointCloud`. Plain functions, not a class --
they hold no state, cache nothing, and need no configuration; they
only adapt an existing model to `CoordinateTransformer`, same
precedent as `terrain.conversion.pointcloud_to_points`.

Contract
--------
- Pure: every function returns a NEW object. The argument passed in
  is never mutated, regardless of whether the underlying type
  happens to be immutable (`SurveyPointSet`/`Feature` are frozen
  dataclasses; `FeatureCollection`/`PointCloud` are not, but are
  still never mutated here).
- Every function takes only `(model, transformer)` -- no
  `source_crs`/`target_crs` parameters. That decision already lives
  inside the `CoordinateTransformer` the caller built (via
  `CoordinateTransformer(source, target)` or
  `CoordinateTransformer.from_operation(...)`); these functions
  never construct one themselves, matching `DXFExporter.export()`/
  `GeoPackageExporter.export()`, which likewise receive prepared
  objects instead of deciding on the caller's behalf.
- `SurveyPointSet`/`FeatureCollection` never gain a `crs` field --
  see Regla G-001 (models don't carry CRS unless it's intrinsic to
  their meaning). `PointCloud` already has `metadata.crs`
  (pre-existing, not introduced here) -- `transform_point_cloud()`
  keeps it consistent with the result rather than leaving it stale.
- `transform_point_cloud()` never calls `PointCloud.clone()` -- see
  TD-003 (`clone()` advertises a deep copy but shares `Chunk`
  objects/arrays with the original). Builds independent `Chunk`s
  directly instead.
- Never touches `topocore.workflow` -- these are geodesy-side
  utilities Workflow's own callers can reach for outside the
  orchestrator, not a new Workflow capability.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from topocore.features.models import FeatureCollection
from topocore.geodesy.transformer import CoordinateTransformer
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.models import SurveyPoint, SurveyPointSet


def transform_survey(survey: SurveyPointSet, transformer: CoordinateTransformer) -> SurveyPointSet:
    """
    Transform every point in `survey` through `transformer`, always
    passing `z` explicitly (`SurveyPoint.z` is required, never
    optional) -- returns a new `SurveyPointSet`, `survey` itself is
    untouched.
    """
    new_points = []
    for point in survey:
        x, y, z = transformer.transform_point(point.x, point.y, point.z)
        new_points.append(
            SurveyPoint(
                id=point.id,
                x=x,
                y=y,
                z=z if z is not None else point.z,
                code=point.code,
            )
        )
    return SurveyPointSet(points=tuple(new_points))


def transform_feature_collection(
    collection: FeatureCollection, transformer: CoordinateTransformer
) -> FeatureCollection:
    """
    Transform every `Feature`'s geometry vertices through
    `transformer` -- returns a new `FeatureCollection`; `collection`
    itself is untouched, even though `FeatureCollection` is
    mutable (`.add()`/`.extend()`/`.normalize_ids()` mutate it
    in-place by design -- this function deliberately doesn't,
    matching every other geodesy transform here). `faces` (MESH
    triangle indices into `vertices`) pass through unchanged --
    indices, not coordinates.

    `FeatureGeometry.__post_init__` re-validates the transformed
    vertices automatically (finite, correct shape, minimum vertex
    count) -- a transform that somehow produced non-finite output
    fails loudly here, not silently downstream.
    """
    new_features = []
    for feature in collection:
        vertices = feature.geometry.vertices
        xs, ys, zs = transformer.transform_array(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        new_vertices = np.column_stack([xs, ys, zs if zs is not None else vertices[:, 2]])

        new_geometry = replace(feature.geometry, vertices=new_vertices)
        new_features.append(replace(feature, geometry=new_geometry))

    return FeatureCollection(features=new_features)


def transform_point_cloud(cloud: PointCloud, transformer: CoordinateTransformer) -> PointCloud:
    """
    Transform every chunk's X/Y/Z through `transformer` -- returns a
    new `PointCloud` with brand-new `Chunk` objects (never
    `PointCloud.clone()` -- see TD-003), copying every other
    attribute's values unchanged. `metadata` on the result mirrors
    `cloud.metadata` field-for-field (never just `crs` in isolation),
    with `crs` updated to `transformer.target_crs` and `bounds`
    recomputed for the new coordinates.

    Every chunk is guaranteed to have X, Y, and Z -- as of TD-004,
    `Chunk.__init__` itself rejects construction if any of them is
    missing, so there is no longer a per-chunk check to make here.
    """
    new_cloud = PointCloud()

    for chunk in cloud:
        new_chunk = Chunk(size=chunk.size, attributes=chunk.attributes, source_id=chunk.source_id)

        xs, ys, zs = transformer.transform_array(
            chunk[PointAttribute.X],
            chunk[PointAttribute.Y],
            chunk[PointAttribute.Z],
        )
        new_chunk[PointAttribute.X][:] = xs
        new_chunk[PointAttribute.Y][:] = ys
        assert zs is not None  # z was passed in above, so transform_array always returns one back
        new_chunk[PointAttribute.Z][:] = zs

        for attribute in chunk.attributes:
            if attribute in (PointAttribute.X, PointAttribute.Y, PointAttribute.Z):
                continue
            new_chunk[attribute][:] = chunk[attribute]

        new_cloud.add_chunk(new_chunk)

    target_crs = transformer.target_crs
    # PointCloud.crs's setter is typed to accept only `str` (even
    # though the underlying field and getter are `str | None`) --
    # never pass None through it. Falls back to the CRS's own name
    # when it has no EPSG code, the same convention CRS.__repr__
    # already uses for a CRS without one.
    new_crs_value = f"EPSG:{target_crs.epsg}" if target_crs.epsg is not None else target_crs.name

    # PointCloud exposes no public setter for the whole `metadata`
    # object (nor individually for source_format/is_georeferenced --
    # only `.crs` has one), so preserving the rest of `cloud.metadata`
    # field-for-field means reaching into `_metadata` directly here.
    # `new_cloud` was just constructed by this function, though, not
    # handed in by the caller -- this finishes its construction
    # rather than reaching into another module's already-live object.
    new_cloud._metadata = replace(cloud.metadata, crs=new_crs_value, bounds=None)
    new_cloud.update_bounds()
    return new_cloud


__all__ = ["transform_feature_collection", "transform_point_cloud", "transform_survey"]
