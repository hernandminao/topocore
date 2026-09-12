"""
topocore.geodesy.georeferencing.apply
========================================

Applying an already-fitted `GeoreferencingResult` (or, more
precisely, its own `HelmertParameters`) to real artifact data --
`apply_helmert_3d_array()` (vectorized, for `PointCloud`/
`FeatureCollection`, which can have very many points) and the 3
per-artifact-type adapters (`apply_georeferencing_to_point_cloud()`,
`apply_georeferencing_to_survey()`,
`apply_georeferencing_to_feature_collection()`).

None of these recompute anything -- they exist purely to apply
parameters `fit_georeferencing()` already produced, following the
exact same "one mathematical transformation, three thin adapters"
principle already established for `topocore.geodesy.transform`'s own
CRS-based transform functions. Kept in `georeferencing/`, not
`geodesy/transform.py`: those functions require a
`CoordinateTransformer` (CRS -> CRS); these apply a `HelmertParameters`
directly in Cartesian space (local -> target), the same distinction
maintained throughout this whole capability.

`apply_helmert_3d_array()` is `apply_helmert_3d()`'s own formula,
vectorized
-----------------------------------------------------------------------
Confirmed directly: `topocore.geodesy.transform.transform_point_cloud()`
and `transform_feature_collection()` both use
`CoordinateTransformer.transform_array()` (NumPy arrays for X/Y/Z
all at once), never a per-point Python loop -- point clouds and
feature vertex sets can have many millions of points, where a
per-point loop would be prohibitively slow. `apply_helmert_3d()`
itself (in `helmert_fit.py`) only ever operates on one point at a
time (used there for control points, always few), so it is not
reused directly for those 2 artifact types here -- this module's own
`apply_helmert_3d_array()` is the exact same position-vector formula,
operating on NumPy arrays instead, verified directly to produce
identical results to calling `apply_helmert_3d()` per point.
`SurveyPointSet` (typically a small number of points) reuses
`apply_helmert_3d()` directly instead, matching
`transform_survey()`'s own per-point loop precedent.

CRS assignment follows each artifact's own existing, confirmed
convention
--------------------------------------------------------------------------
- `PointCloud`: `crs` is a `str`, set by replacing `.metadata`
  (`PointCloud` exposes no public setter for the whole metadata
  object), then `update_bounds()` is called -- the exact pattern
  `transform_point_cloud()` itself already uses.
- `SurveyPointSet`: `crs` is a real `CRS` object, set once at
  construction (the class is frozen).
- `FeatureCollection`: `crs` is a `str`, a plain mutable field.

`target_crs: CRS` is a parameter of each adapter here, not a field of
`GeoreferencingResult` -- that contract is already frozen and
deliberately not extended (see this capability's own design
decisions elsewhere): `fit_georeferencing()` has no opinion on what
CRS the fitted transformation's own target coordinates should be
declared as, since it never receives one -- that is exclusively
`Workflow.georeference()`'s own concern, supplied by the caller.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import replace
from math import radians

import numpy as np
from numpy.typing import NDArray

from topocore.core.types import FloatArray1D
from topocore.features.models import FeatureCollection
from topocore.geodesy.crs import CRS
from topocore.geodesy.georeferencing.helmert_fit import apply_helmert_3d
from topocore.geodesy.helmert import HelmertParameters
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.models import SurveyPoint, SurveyPointSet


def _crs_to_string(target_crs: CRS) -> str:
    # Same "EPSG:{code} or bare .name" convention already established
    # throughout this codebase (transform_point_cloud(),
    # transform_feature_collection(), etc.) -- not reinvented here.
    return f"EPSG:{target_crs.epsg}" if target_crs.epsg is not None else target_crs.name


def apply_helmert_3d_array(
    xs: FloatArray1D,
    ys: FloatArray1D,
    zs: FloatArray1D,
    parameters: HelmertParameters,
    *,
    apply_scale_to_z: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Vectorized `apply_helmert_3d()` -- the exact same position-vector
    formula (see `helmert_fit.py`'s own top-level docstring), applied
    to NumPy arrays instead of a single point. Verified directly to
    produce identical results to calling `apply_helmert_3d()` once
    per point, `apply_scale_to_z` included -- see that function's own
    docstring for the full rationale (in short: pass `False`
    specifically for a `HELMERT_2D`-derived `parameters`, since that
    strategy's own model never included a scale term on Z at all).
    """
    s = parameters.scale * 1e-6
    rx = radians(parameters.rx / 3600.0)
    ry = radians(parameters.ry / 3600.0)
    rz = radians(parameters.rz / 3600.0)

    xt = xs + parameters.tx + s * xs - rz * ys + ry * zs
    yt = ys + parameters.ty + s * ys + rz * xs - rx * zs
    z_scale_term = s * zs if apply_scale_to_z else 0.0
    zt = zs + parameters.tz + z_scale_term - ry * xs + rx * ys
    return xt, yt, zt


def apply_georeferencing_to_point_cloud(
    cloud: PointCloud, parameters: HelmertParameters, target_crs: CRS, *, apply_scale_to_z: bool = True
) -> PointCloud:
    """
    Returns a new `PointCloud` (never mutates `cloud`) with every
    chunk's X/Y/Z transformed via `parameters`; every other attribute
    is copied unchanged. Follows
    `topocore.geodesy.transform.transform_point_cloud()`'s own exact
    pattern: brand-new `Chunk` objects (never `PointCloud.clone()`),
    `metadata` mirrored field-for-field with `crs` updated and
    `bounds` recomputed via `update_bounds()`.

    `apply_scale_to_z` defaults to `True` -- pass `False` when
    `parameters` came from a `HELMERT_2D` fit (see
    `apply_helmert_3d_array()`'s own docstring for why this matters).
    """
    new_cloud = PointCloud()

    for chunk in cloud:
        new_chunk = Chunk(size=chunk.size, attributes=chunk.attributes, source_id=chunk.source_id)

        xs, ys, zs = apply_helmert_3d_array(
            chunk[PointAttribute.X],
            chunk[PointAttribute.Y],
            chunk[PointAttribute.Z],
            parameters,
            apply_scale_to_z=apply_scale_to_z,
        )
        new_chunk[PointAttribute.X][:] = xs
        new_chunk[PointAttribute.Y][:] = ys
        new_chunk[PointAttribute.Z][:] = zs

        for attribute in chunk.attributes:
            if attribute in (PointAttribute.X, PointAttribute.Y, PointAttribute.Z):
                continue
            new_chunk[attribute][:] = chunk[attribute]

        new_cloud.add_chunk(new_chunk)

    new_cloud._metadata = replace(cloud.metadata, crs=_crs_to_string(target_crs), bounds=None)
    new_cloud.update_bounds()
    return new_cloud


def apply_georeferencing_to_survey(
    survey: SurveyPointSet, parameters: HelmertParameters, target_crs: CRS, *, apply_scale_to_z: bool = True
) -> SurveyPointSet:
    """
    Returns a new `SurveyPointSet` (never mutates `survey` -- it is
    frozen already) with every point's X/Y/Z transformed via
    `parameters`; `id`/`code` are carried over unchanged. `crs` on
    the result is `target_crs` directly (a real `CRS` object,
    matching `SurveyPointSet.crs`'s own type -- unlike `PointCloud`/
    `FeatureCollection`, no string conversion is needed here).

    `apply_scale_to_z` defaults to `True` -- pass `False` when
    `parameters` came from a `HELMERT_2D` fit (see
    `apply_helmert_3d()`'s own docstring for why this matters).
    """
    new_points = []
    for point in survey:
        x, y, z = apply_helmert_3d((point.x, point.y, point.z), parameters, apply_scale_to_z=apply_scale_to_z)
        new_points.append(SurveyPoint(id=point.id, x=x, y=y, z=z, code=point.code))

    return SurveyPointSet(points=tuple(new_points), crs=target_crs)


def apply_georeferencing_to_feature_collection(
    collection: FeatureCollection, parameters: HelmertParameters, target_crs: CRS, *, apply_scale_to_z: bool = True
) -> FeatureCollection:
    """
    Returns a new `FeatureCollection` (never mutates `collection`,
    even though it is itself mutable -- matching
    `topocore.geodesy.transform.transform_feature_collection()`'s own
    deliberate choice) with every `Feature`'s geometry vertices
    transformed via `parameters`; `faces` (MESH triangle indices)
    pass through unchanged -- indices, not coordinates. `crs` on the
    result is set from `target_crs`, converted to the same string
    convention `FeatureCollection.crs` already uses.

    `apply_scale_to_z` defaults to `True` -- pass `False` when
    `parameters` came from a `HELMERT_2D` fit (see
    `apply_helmert_3d_array()`'s own docstring for why this matters).
    """
    new_features = []
    for feature in collection:
        vertices = feature.geometry.vertices
        xs, ys, zs = apply_helmert_3d_array(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], parameters, apply_scale_to_z=apply_scale_to_z
        )
        new_vertices = np.column_stack([xs, ys, zs])

        new_geometry = replace(feature.geometry, vertices=new_vertices)
        new_features.append(replace(feature, geometry=new_geometry))

    return FeatureCollection(features=new_features, crs=_crs_to_string(target_crs))


__all__ = [
    "apply_georeferencing_to_feature_collection",
    "apply_georeferencing_to_point_cloud",
    "apply_georeferencing_to_survey",
    "apply_helmert_3d_array",
]
