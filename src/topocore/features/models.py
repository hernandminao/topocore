"""
topocore.features.models
==========================

Core data models for extracted geospatial features.

Separates pure geometry (`FeatureGeometry`) from the semantic
record (`Feature`) that wraps it with a type, category, confidence,
provenance metadata, and free-form attributes. `FeatureCollection`
is the aggregate result type that detectors and the manager return,
and that PR16 (DXF) / PR17 (GeoPackage) will consume for export.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from topocore.core.types import FloatArray1D, IntArray1D
from topocore.features.exceptions import GeometryError


class GeometryType(StrEnum):
    """
    Supported feature geometry primitives.

    Named to map directly onto the eventual DXF/GeoPackage export
    targets: POINT -> DXF POINT / GPKG Point, POLYLINE -> DXF
    (LW)POLYLINE / GPKG LineString, POLYGON -> GPKG Polygon (closed
    POLYLINE in DXF), MESH -> DXF 3DFACE/MESH (no direct GPKG
    equivalent; meshes are exported as a set of triangles).
    """

    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    MESH = "mesh"


class FeatureCategory(StrEnum):
    """Top-level grouping, one per `features/` subpackage."""

    TERRAIN = "terrain"
    BUILDING = "building"
    INFRASTRUCTURE = "infrastructure"
    DRAINAGE = "drainage"
    VEGETATION = "vegetation"
    UTILITY = "utility"


class FeatureType(StrEnum):
    """
    Specific entity types, grouped by category in comments.

    Single source of truth for "what TopoCore can detect" — new
    detectors add a member here rather than inventing ad hoc string
    literals.
    """

    # terrain
    BREAKLINE = "breakline"
    CONTOUR = "contour"
    EMBANKMENT = "embankment"
    SLOPE_CHANGE = "slope_change"
    # building
    BUILDING = "building"
    WALL = "wall"
    RETAINING_WALL = "retaining_wall"
    ROOF = "roof"
    # infrastructure
    ROAD = "road"
    CURB = "curb"
    PARKING = "parking"
    DRIVEWAY = "driveway"
    # drainage
    DRAINAGE = "drainage"
    CHANNEL = "channel"
    MANHOLE = "manhole"
    INSPECTION_CHAMBER = "inspection_chamber"
    # vegetation
    TREE = "tree"
    SHRUB = "shrub"
    GRASS = "grass"
    # utility
    POLE = "pole"
    SIGN = "sign"
    LIGHT_POLE = "light_pole"


class ContextField(StrEnum):
    """
    Every field `DetectionContext` (see `protocols.py`) can carry.

    Detectors declare `required_inputs` using these members instead
    of raw strings, so a typo like `"pointcloud"` vs `"point_cloud"`
    is caught by mypy/IDE autocomplete instead of surfacing as a
    silently-always-missing input at runtime.

    Member *values* match the corresponding `DetectionContext`
    attribute name exactly, since `BaseFeatureDetector.detect()`
    resolves required inputs via ``getattr(context, field.value)``.
    Member *names* are the readable, autocomplete-friendly form —
    e.g. ``ContextField.POINT_CLOUD.value == "cloud"``, matching the
    short attribute name already used elsewhere in TopoCore (e.g.
    ``MachineLearningClassifier.fit(self, cloud, ...)``).
    """

    POINT_CLOUD = "cloud"
    CLASSIFICATION = "classification"
    NORMALS = "normals"
    PCA_FEATURES = "pca_features"
    TIN = "tin"
    DTM = "dtm"
    SLOPE = "slope"


# Which geometry primitives are valid for each feature type. A
# frozenset (rather than a single GeometryType) leaves room for
# feature types that can legitimately be represented more than one
# way — e.g. a TREE as a trunk POINT or a canopy POLYGON, depending
# on what the source data supports.
_EXPECTED_GEOMETRY: dict[FeatureType, frozenset[GeometryType]] = {
    FeatureType.BREAKLINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.CONTOUR: frozenset({GeometryType.POLYLINE}),
    FeatureType.EMBANKMENT: frozenset({GeometryType.POLYLINE}),
    FeatureType.SLOPE_CHANGE: frozenset({GeometryType.POLYLINE}),
    FeatureType.BUILDING: frozenset({GeometryType.POLYGON}),
    FeatureType.WALL: frozenset({GeometryType.POLYLINE}),
    FeatureType.RETAINING_WALL: frozenset({GeometryType.POLYLINE}),
    FeatureType.ROOF: frozenset({GeometryType.MESH}),
    FeatureType.ROAD: frozenset({GeometryType.POLYGON}),
    FeatureType.CURB: frozenset({GeometryType.POLYLINE}),
    FeatureType.PARKING: frozenset({GeometryType.POLYGON}),
    FeatureType.DRIVEWAY: frozenset({GeometryType.POLYGON}),
    FeatureType.DRAINAGE: frozenset({GeometryType.POLYLINE}),
    FeatureType.CHANNEL: frozenset({GeometryType.POLYLINE}),
    FeatureType.MANHOLE: frozenset({GeometryType.POINT}),
    FeatureType.INSPECTION_CHAMBER: frozenset({GeometryType.POINT}),
    FeatureType.TREE: frozenset({GeometryType.POINT, GeometryType.POLYGON}),
    FeatureType.SHRUB: frozenset({GeometryType.POINT, GeometryType.POLYGON}),
    FeatureType.GRASS: frozenset({GeometryType.POLYGON}),
    FeatureType.POLE: frozenset({GeometryType.POINT}),
    FeatureType.SIGN: frozenset({GeometryType.POINT}),
    FeatureType.LIGHT_POLE: frozenset({GeometryType.POINT}),
}

_MIN_VERTICES: dict[GeometryType, int] = {
    GeometryType.POINT: 1,
    GeometryType.POLYLINE: 2,
    GeometryType.POLYGON: 3,
    GeometryType.MESH: 3,
}


@dataclass(frozen=True, slots=True)
class FeatureGeometry:
    """
    Immutable geometric representation of a feature.

    Parameters
    ----------
    geometry_type
        The geometric primitive this represents.
    vertices
        Vertex coordinates as an ``(n, 3)`` array, always XYZ (use
        0.0 for Z on purely 2D features rather than a separate 2D
        code path, so downstream export always sees a consistent
        shape).
    closed
        Whether a POLYGON/POLYLINE's last vertex implicitly connects
        back to the first. Ignored for POINT/MESH.
    faces
        For MESH only: ``(m, 3)`` array of vertex-index triplets
        defining triangles. ``None`` for all other geometry types.

    Raises
    ------
    GeometryError
        If the vertex array shape is invalid, contains non-finite
        coordinates, or has fewer vertices than the geometry type
        requires.
    """

    geometry_type: GeometryType
    vertices: NDArray[np.float64]
    closed: bool = False
    faces: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise GeometryError(f"Vertices must have shape (n, 3); got {self.vertices.shape}.")

        min_vertices = _MIN_VERTICES[self.geometry_type]
        if self.vertices.shape[0] < min_vertices:
            raise GeometryError(
                f"{self.geometry_type.value} requires at least {min_vertices} vertices; got {self.vertices.shape[0]}."
            )

        if not np.all(np.isfinite(self.vertices)):
            raise GeometryError("Vertices must contain only finite coordinates.")

        if self.geometry_type == GeometryType.MESH:
            if self.faces is None:
                raise GeometryError("MESH geometry requires `faces`.")
            if self.faces.ndim != 2 or self.faces.shape[1] != 3:
                raise GeometryError(f"Faces must have shape (m, 3); got {self.faces.shape}.")
            if self.faces.size > 0 and (int(self.faces.min()) < 0 or int(self.faces.max()) >= self.vertices.shape[0]):
                raise GeometryError("Face indices out of range of `vertices`.")
        elif self.faces is not None:
            raise GeometryError(f"`faces` is only valid for MESH geometry, not {self.geometry_type.value}.")

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """``(min_x, min_y, min_z, max_x, max_y, max_z)``."""
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        return (
            float(mins[0]),
            float(mins[1]),
            float(mins[2]),
            float(maxs[0]),
            float(maxs[1]),
            float(maxs[2]),
        )


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """
    Provenance for a detected feature — QA, auditing (PR19), and
    comparing detector algorithms/versions.

    Kept separate from `Feature.confidence`, which stays a
    first-class field on `Feature` itself; `metadata` is for
    information about *how* the feature was produced, not the
    detection score itself.

    Parameters
    ----------
    detector
        ``name()`` of the detector that produced this feature.
    version
        Detector algorithm version string, for reproducibility when
        comparing runs across TopoCore releases.
    inputs_used
        Which `ContextField`s the detector actually consumed to
        produce this specific feature (may be a subset of the
        detector's `required_inputs`, if some were only used for
        validation rather than the algorithm itself).
    extra
        Any additional detector-specific provenance not covered by
        the fields above.
    """

    detector: str
    version: str = "1.0"
    inputs_used: frozenset[ContextField] = field(default_factory=frozenset)
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.extra, MappingProxyType):
            object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


@dataclass(frozen=True, slots=True)
class Feature:
    """
    A single detected geospatial feature.

    Parameters
    ----------
    feature_id
        Identifier within a `FeatureCollection`. May be ``None`` or
        a detector-local (possibly colliding-with-other-detectors)
        value while a detector is building its own
        `FeatureCollection`; `FeatureExtractionManager` always calls
        `FeatureCollection.normalize_ids()` before returning, which
        overwrites this with a contiguous, collision-free ``1..N``
        range. Only features obtained by calling a detector directly
        (bypassing the manager) may retain a non-normalized value.
    category
        Top-level grouping (matches the `features/` subpackage that
        produced it).
    feature_type
        Specific entity type.
    geometry
        The feature's shape.
    confidence
        Detector confidence in ``[0, 1]``. Rule-based detectors that
        don't naturally produce a probability should report ``1.0``.
    metadata
        Provenance information (which detector/version/inputs
        produced this feature). ``None`` is allowed for lightweight
        or test-constructed features, but detectors are expected to
        always populate it — see `BaseFeatureDetector._metadata()`.
    attributes
        Free-form semantic attributes (e.g. ``{"diameter_m": 0.4}``
        for a manhole, ``{"height_m": 12.3}`` for a building). Kept
        as a generic mapping rather than per-type dataclasses so new
        attributes don't require a schema migration; detector
        modules are expected to document the attribute keys they
        populate.
    source_point_indices
        Indices into the source point cloud that contributed to this
        feature, for traceability and later refinement. ``None`` if
        the feature was derived from terrain geometry (TIN/DTM)
        rather than directly from points.

    Raises
    ------
    GeometryError
        If ``geometry.geometry_type`` isn't one of the primitives
        allowed for ``feature_type`` (see `_EXPECTED_GEOMETRY`), or
        if ``confidence`` is outside ``[0, 1]``.
    """

    feature_id: int | None
    category: FeatureCategory
    feature_type: FeatureType
    geometry: FeatureGeometry
    confidence: float = 1.0
    metadata: FeatureMetadata | None = None
    attributes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    source_point_indices: IntArray1D | None = None

    def __post_init__(self) -> None:
        allowed = _EXPECTED_GEOMETRY.get(self.feature_type)
        if allowed is not None and self.geometry.geometry_type not in allowed:
            allowed_names = ", ".join(sorted(g.value for g in allowed))
            raise GeometryError(
                f"{self.feature_type.value} expects one of [{allowed_names}] geometry, "
                f"got {self.geometry.geometry_type.value}."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise GeometryError(f"confidence must be in [0, 1]; got {self.confidence}.")

        if not isinstance(self.attributes, MappingProxyType):
            object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(slots=True)
class FeatureCollection:
    """
    Aggregate result of one or more detectors.

    Parameters
    ----------
    features
        All detected features, in no particular order.
    """

    features: list[Feature] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.features)

    def add(self, feature: Feature) -> None:
        self.features.append(feature)

    def extend(self, other: FeatureCollection) -> None:
        self.features.extend(other.features)

    def by_type(self, feature_type: FeatureType) -> list[Feature]:
        return [f for f in self.features if f.feature_type == feature_type]

    def by_category(self, category: FeatureCategory) -> list[Feature]:
        return [f for f in self.features if f.category == category]

    def confidence_array(self) -> FloatArray1D:
        return np.array([f.confidence for f in self.features], dtype=np.float64)

    def normalize_ids(self) -> None:
        """
        Reassign ``feature_id`` for every feature to a contiguous
        ``1..N`` range, in the collection's current order.

        Since `Feature` is frozen, this replaces each entry with
        ``dataclasses.replace(feature, feature_id=...)`` rather than
        mutating in place. Called automatically by
        `FeatureExtractionManager` after detection, so any detector
        -local IDs (which may collide across detectors) are only
        meaningful before this runs.
        """
        self.features = [replace(f, feature_id=i) for i, f in enumerate(self.features, start=1)]

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float] | None:
        """Combined XYZ bounds of every feature, or ``None`` if empty."""
        if not self.features:
            return None

        all_bounds = np.array([f.geometry.bounds for f in self.features])
        mins = all_bounds[:, :3].min(axis=0)
        maxs = all_bounds[:, 3:].max(axis=0)
        return (
            float(mins[0]),
            float(mins[1]),
            float(mins[2]),
            float(maxs[0]),
            float(maxs[1]),
            float(maxs[2]),
        )


__all__ = [
    "GeometryType",
    "FeatureCategory",
    "FeatureType",
    "ContextField",
    "FeatureGeometry",
    "FeatureMetadata",
    "Feature",
    "FeatureCollection",
]
