"""
topocore.geodesy.vertical.transform
======================================

Applies a `VerticalTransformer` to TopoCore's own models -- the
vertical-axis counterpart to `topocore.geodesy.transform`'s own
`transform_point_cloud`/`transform_survey`/
`transform_feature_collection`, kept in this separate module rather
than added to that one, matching the deliberate horizontal/vertical
separation established throughout `topocore.geodesy.vertical`.

Caller responsibility, stated plainly: `VerticalTransformer` expects
`(longitude, latitude)` to look up the geoid undulation at that
point. **These functions assume the model's own X/Y are already
geographic degrees in the same CRS the geoid grid itself is defined
in.** None of these functions validate this -- confirmed throughout
this project's own geodesy audit, `PointCloud`/`SurveyPointSet`/
`FeatureCollection` do not reliably carry a CRS at all (see
`../../14-geodesy/io-boundary.md`), so there is nothing here to check
this against. If your data's X/Y are in a projected CRS, transform to
a geographic one first (`topocore.geodesy.transform_point_cloud()`
and friends) before applying a vertical transform.

Fail-whole-operation, not partial-silent
-------------------------------------------
If any single point/vertex in the input falls outside the geoid
grid's own extent (or lands on a nodata cell), the entire function
call raises `MissingGeoidGridError` immediately -- it does not
silently skip that point, and does not return a partially-corrected
result where some heights were shifted and others were not. A
partially-corrected `PointCloud` (some points genuinely orthometric,
others still ellipsoidal, with no marker distinguishing which is
which) would be a worse outcome than a clear, whole-operation
failure -- exactly the kind of silent, misleading result this
package exists to prevent.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import dataclasses

from topocore.features.models import FeatureCollection
from topocore.geodesy.vertical.transformer import VerticalTransformer
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.models import SurveyPoint, SurveyPointSet


def transform_point_cloud_vertical(cloud: PointCloud, transformer: VerticalTransformer) -> PointCloud:
    """
    Returns a new `PointCloud` with `Z` shifted via
    `transformer.ellipsoidal_to_orthometric()`. `X`/`Y` and every
    other attribute are copied through unchanged -- this function
    only ever touches `Z`. The input is never mutated.
    """
    new_cloud = PointCloud()

    for chunk in cloud:
        x_values = chunk[PointAttribute.X]
        y_values = chunk[PointAttribute.Y]
        z_values = chunk[PointAttribute.Z]

        new_chunk = Chunk(
            size=chunk.size,
            attributes=list(chunk.attributes),
            source_id=chunk.source_id,
        )

        for attribute in chunk.attributes:
            if attribute is PointAttribute.Z:
                continue
            new_chunk[attribute][:] = chunk[attribute]

        new_chunk[PointAttribute.Z][:] = [
            transformer.ellipsoidal_to_orthometric(float(x), float(y), float(z))
            for x, y, z in zip(x_values, y_values, z_values, strict=True)
        ]

        new_cloud.add_chunk(new_chunk)

    return new_cloud


def transform_survey_vertical(survey: SurveyPointSet, transformer: VerticalTransformer) -> SurveyPointSet:
    """
    Returns a new `SurveyPointSet` with every point's `z` shifted via
    `transformer.ellipsoidal_to_orthometric()`. `id`/`x`/`y`/`code`
    are carried over unchanged. `crs` is preserved unchanged from the
    input -- a vertical-only shift never changes what horizontal CRS
    `x`/`y` are already expressed in, so artificially resetting it
    (which the default `SurveyPointSet(points=...)` constructor would
    otherwise silently do, since `crs` defaults to `None`) would
    incorrectly discard a real, already-known CRS. The input is never
    mutated (`SurveyPoint` is already frozen).
    """
    new_points = tuple(
        SurveyPoint(
            id=point.id,
            x=point.x,
            y=point.y,
            z=transformer.ellipsoidal_to_orthometric(point.x, point.y, point.z),
            code=point.code,
        )
        for point in survey
    )
    return SurveyPointSet(points=new_points, crs=survey.crs)


def transform_feature_collection_vertical(
    collection: FeatureCollection, transformer: VerticalTransformer
) -> FeatureCollection:
    """
    Returns a new `FeatureCollection` with every `Feature`'s geometry
    vertices' `Z` shifted via
    `transformer.ellipsoidal_to_orthometric()`. `X`/`Y` (per vertex)
    and every other field (`faces`, `attributes`, `feature_id`, ...)
    are carried over unchanged. `FeatureGeometry.__post_init__`
    re-validates the result automatically, same as
    `topocore.geodesy.transform_feature_collection()`. The input is
    never mutated.

    `crs` is preserved unchanged from `collection` -- a vertical-only
    shift never changes what horizontal CRS `X`/`Y` are already
    expressed in, so leaving `FeatureCollection()`'s own default
    (`crs=None`) in place here would incorrectly discard a real,
    already-known CRS.
    """
    new_collection = FeatureCollection(crs=collection.crs)

    for feature in collection:
        vertices = feature.geometry.vertices
        new_vertices = vertices.copy()

        for index in range(len(vertices)):
            x, y, z = vertices[index]
            new_vertices[index, 2] = transformer.ellipsoidal_to_orthometric(float(x), float(y), float(z))

        new_geometry = dataclasses.replace(feature.geometry, vertices=new_vertices)
        new_collection.add(dataclasses.replace(feature, geometry=new_geometry))

    return new_collection


__all__ = [
    "transform_feature_collection_vertical",
    "transform_point_cloud_vertical",
    "transform_survey_vertical",
]
