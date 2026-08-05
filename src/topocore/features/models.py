"""
topocore.features.models
==========================

Core data models for extracted geospatial features.

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
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    MESH = "mesh"


class FeatureCategory(StrEnum):
    """Top-level semantic domain of a feature."""

    TERRAIN = "terrain"
    BUILDING = "building"
    INFRASTRUCTURE = "infrastructure"
    DRAINAGE = "drainage"
    VEGETATION = "vegetation"
    UTILITY = "utility"
    CADASTRE = "cadastre"
    CONTROL = "control"


class FeatureType(StrEnum):
    """
    Every semantic feature type TopoCore can represent -- produced
    either by automatic PR15 detection (LiDAR/TIN) or by field
    survey code interpretation (total station/GNSS via
    ``topocore.features.feature_codes``). Single source of truth for
    "what TopoCore can represent", not "what a detector can find".
    """

    # --- terrain (PR15 detectors) ---
    BREAKLINE = "breakline"
    CONTOUR = "contour"
    SLOPE_CHANGE = "slope_change"
    EMBANKMENT = "embankment"
    EMBANKMENT_CREST = "embankment_crest"
    EMBANKMENT_TOE = "embankment_toe"

    # --- building ---
    BUILDING = "building"
    WALL = "wall"
    RETAINING_WALL = "retaining_wall"
    ROOF = "roof"
    ROOF_EDGE = "roof_edge"
    FENCE = "fence"
    GATE = "gate"
    HARDSCAPE = "hardscape"
    POOL = "pool"
    TANK = "tank"
    STRUCTURAL_ELEMENT = "structural_element"
    STAIR = "stair"
    RAMP = "ramp"
    OPENING = "opening"

    # --- infrastructure ---
    ROAD = "road"
    CURB = "curb"
    PARKING = "parking"
    DRIVEWAY = "driveway"
    CENTERLINE = "centerline"
    PAVEMENT_EDGE = "pavement_edge"
    SIDEWALK = "sidewalk"
    MEDIAN = "median"
    SHOULDER = "shoulder"
    GUARDRAIL = "guardrail"
    BRIDGE = "bridge"
    TUNNEL = "tunnel"
    TRAFFIC_ISLAND = "traffic_island"
    ROUNDABOUT = "roundabout"

    # --- drainage ---
    DRAINAGE = "drainage"
    CHANNEL = "channel"
    MANHOLE = "manhole"
    INSPECTION_CHAMBER = "inspection_chamber"
    WATERCOURSE = "watercourse"
    WATERBODY = "waterbody"
    SEWER_LINE = "sewer_line"
    STORM_DRAIN_INLET = "storm_drain_inlet"
    SPILLWAY = "spillway"
    DIKE = "dike"
    DAM = "dam"

    # --- vegetation ---
    TREE = "tree"
    SHRUB = "shrub"
    GRASS = "grass"
    FOREST = "forest"
    CROP = "crop"
    VEGETATION_LINE = "vegetation_line"
    TREE_TRUNK = "tree_trunk"
    TREE_STUMP = "tree_stump"
    TREE_ROOT = "tree_root"

    # --- utility ---
    POLE = "pole"
    SIGN = "sign"
    LIGHT_POLE = "light_pole"
    TRANSMISSION_TOWER = "transmission_tower"
    TRANSFORMER = "transformer"
    SUBSTATION = "substation"
    POWER_LINE = "power_line"
    TELECOM_LINE = "telecom_line"
    ANTENNA = "antenna"
    WATER_LINE = "water_line"
    VALVE = "valve"
    HYDRANT = "hydrant"
    WATER_METER = "water_meter"
    GAS_LINE = "gas_line"
    GAS_REGULATOR = "gas_regulator"
    UTILITY_LINE = "utility_line"
    UTILITY_BOX = "utility_box"
    UTILITY_CHAMBER = "utility_chamber"

    # --- cadastre ---
    PARCEL = "parcel"
    BLOCK = "block"
    ZONE = "zone"
    RIGHT_OF_WAY = "right_of_way"
    BOUNDARY = "boundary"
    EASEMENT = "easement"
    SETBACK = "setback"
    BOUNDARY_MONUMENT = "boundary_monument"
    ADMINISTRATIVE_BOUNDARY = "administrative_boundary"
    REFERENCE_POINT = "reference_point"

    # --- control ---
    CONTROL_POINT = "control_point"
    ALIGNMENT_PI = "alignment_pi"


class ContextField(StrEnum):
    POINT_CLOUD = "cloud"
    CLASSIFICATION = "classification"
    NORMALS = "normals"
    PCA_FEATURES = "pca_features"
    TIN = "tin"
    DTM = "dtm"
    SLOPE = "slope"


_EXPECTED_GEOMETRY: dict[FeatureType, frozenset[GeometryType]] = {
    # --- terrain ---
    FeatureType.BREAKLINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.CONTOUR: frozenset({GeometryType.POLYLINE}),
    FeatureType.SLOPE_CHANGE: frozenset({GeometryType.POLYLINE}),
    FeatureType.EMBANKMENT: frozenset({GeometryType.POLYLINE}),
    FeatureType.EMBANKMENT_CREST: frozenset({GeometryType.POLYLINE}),
    FeatureType.EMBANKMENT_TOE: frozenset({GeometryType.POLYLINE}),
    # --- building ---
    FeatureType.BUILDING: frozenset({GeometryType.POLYGON}),
    # WALL/RETAINING_WALL accept both representations: PR15's
    # WallDetector/RetainingWallDetector build a POLYGON footprint
    # (convex_hull_polygon over a vertical-facade point cluster),
    # while a field-surveyed MURO/MURCONT is a POLYLINE (its traced
    # axis). Same physical object, two legitimate representations
    # depending on the data source -- not two different concepts, so
    # this is not split into WALL_LINE.
    FeatureType.WALL: frozenset({GeometryType.POLYLINE, GeometryType.POLYGON}),
    FeatureType.RETAINING_WALL: frozenset({GeometryType.POLYLINE, GeometryType.POLYGON}),
    # CUBIERTA (field-surveyed roof footprint, POLYGON) is the same
    # object as ROOF (PR15's triangulated MESH) with a different
    # representation -- same precedent as WALL. ALERO (roof edge,
    # LINE) is the *boundary* of the roof, not the roof itself --
    # same precedent as ROAD/PAVEMENT_EDGE, so it gets its own type
    # (ROOF_EDGE) rather than being folded in here.
    FeatureType.ROOF: frozenset({GeometryType.MESH, GeometryType.POLYGON}),
    FeatureType.ROOF_EDGE: frozenset({GeometryType.POLYLINE}),
    FeatureType.FENCE: frozenset({GeometryType.POLYLINE}),
    FeatureType.GATE: frozenset({GeometryType.POINT}),
    FeatureType.HARDSCAPE: frozenset({GeometryType.POLYGON}),
    FeatureType.POOL: frozenset({GeometryType.POLYGON}),
    FeatureType.TANK: frozenset({GeometryType.POLYGON}),
    FeatureType.STRUCTURAL_ELEMENT: frozenset({GeometryType.POINT}),
    FeatureType.STAIR: frozenset({GeometryType.POLYLINE}),
    FeatureType.RAMP: frozenset({GeometryType.POLYLINE}),
    FeatureType.OPENING: frozenset({GeometryType.POINT}),
    # --- infrastructure ---
    FeatureType.ROAD: frozenset({GeometryType.POLYGON}),
    FeatureType.CURB: frozenset({GeometryType.POLYLINE}),
    FeatureType.PARKING: frozenset({GeometryType.POLYGON}),
    FeatureType.DRIVEWAY: frozenset({GeometryType.POLYGON}),
    FeatureType.CENTERLINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.PAVEMENT_EDGE: frozenset({GeometryType.POLYLINE}),
    FeatureType.SIDEWALK: frozenset({GeometryType.POLYLINE}),
    FeatureType.MEDIAN: frozenset({GeometryType.POLYLINE}),
    FeatureType.SHOULDER: frozenset({GeometryType.POLYLINE}),
    FeatureType.GUARDRAIL: frozenset({GeometryType.POLYLINE}),
    FeatureType.BRIDGE: frozenset({GeometryType.POLYGON}),
    FeatureType.TUNNEL: frozenset({GeometryType.POLYGON}),
    FeatureType.TRAFFIC_ISLAND: frozenset({GeometryType.POLYGON}),
    FeatureType.ROUNDABOUT: frozenset({GeometryType.POLYGON}),
    # --- drainage ---
    FeatureType.DRAINAGE: frozenset({GeometryType.POLYLINE}),
    FeatureType.CHANNEL: frozenset({GeometryType.POLYLINE}),
    FeatureType.MANHOLE: frozenset({GeometryType.POINT}),
    FeatureType.INSPECTION_CHAMBER: frozenset({GeometryType.POINT}),
    FeatureType.WATERCOURSE: frozenset({GeometryType.POLYLINE}),
    FeatureType.WATERBODY: frozenset({GeometryType.POLYGON}),
    FeatureType.SEWER_LINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.STORM_DRAIN_INLET: frozenset({GeometryType.POINT}),
    FeatureType.SPILLWAY: frozenset({GeometryType.POLYLINE}),
    FeatureType.DIKE: frozenset({GeometryType.POLYLINE}),
    FeatureType.DAM: frozenset({GeometryType.POLYLINE}),
    # --- vegetation ---
    FeatureType.TREE: frozenset({GeometryType.POINT, GeometryType.POLYGON}),
    FeatureType.SHRUB: frozenset({GeometryType.POINT, GeometryType.POLYGON}),
    FeatureType.GRASS: frozenset({GeometryType.POLYGON}),
    FeatureType.FOREST: frozenset({GeometryType.POLYGON}),
    FeatureType.CROP: frozenset({GeometryType.POLYGON}),
    FeatureType.VEGETATION_LINE: frozenset({GeometryType.POLYLINE}),
    # Added when TRONCO/TOCON/RAIZ were migrated out of vegetation.py's
    # deferred set -- ROCAARBOL remains deferred, its semantics unclear
    # from the catalog definition alone ("Tree on Rock" -- marks the
    # rock, the tree, or a composite feature? undetermined).
    FeatureType.TREE_TRUNK: frozenset({GeometryType.POINT}),
    FeatureType.TREE_STUMP: frozenset({GeometryType.POINT}),
    FeatureType.TREE_ROOT: frozenset({GeometryType.POINT}),
    # --- utility ---
    FeatureType.POLE: frozenset({GeometryType.POINT}),
    FeatureType.SIGN: frozenset({GeometryType.POINT}),
    FeatureType.LIGHT_POLE: frozenset({GeometryType.POINT}),
    FeatureType.TRANSMISSION_TOWER: frozenset({GeometryType.POINT}),
    FeatureType.TRANSFORMER: frozenset({GeometryType.POINT}),
    FeatureType.SUBSTATION: frozenset({GeometryType.POLYGON}),
    FeatureType.POWER_LINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.TELECOM_LINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.ANTENNA: frozenset({GeometryType.POINT}),
    FeatureType.WATER_LINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.VALVE: frozenset({GeometryType.POINT}),
    FeatureType.HYDRANT: frozenset({GeometryType.POINT}),
    FeatureType.WATER_METER: frozenset({GeometryType.POINT}),
    FeatureType.GAS_LINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.GAS_REGULATOR: frozenset({GeometryType.POINT}),
    FeatureType.UTILITY_LINE: frozenset({GeometryType.POLYLINE}),
    FeatureType.UTILITY_BOX: frozenset({GeometryType.POINT}),
    FeatureType.UTILITY_CHAMBER: frozenset({GeometryType.POINT}),
    # --- cadastre ---
    FeatureType.PARCEL: frozenset({GeometryType.POLYGON}),
    FeatureType.BLOCK: frozenset({GeometryType.POLYGON}),
    FeatureType.ZONE: frozenset({GeometryType.POLYGON}),
    FeatureType.RIGHT_OF_WAY: frozenset({GeometryType.POLYGON}),
    FeatureType.BOUNDARY: frozenset({GeometryType.POLYLINE}),
    FeatureType.EASEMENT: frozenset({GeometryType.POLYLINE}),
    FeatureType.SETBACK: frozenset({GeometryType.POLYLINE}),
    FeatureType.BOUNDARY_MONUMENT: frozenset({GeometryType.POINT}),
    FeatureType.ADMINISTRATIVE_BOUNDARY: frozenset({GeometryType.POLYGON}),
    FeatureType.REFERENCE_POINT: frozenset({GeometryType.POINT}),
    # --- control ---
    FeatureType.CONTROL_POINT: frozenset({GeometryType.POINT}),
    FeatureType.ALIGNMENT_PI: frozenset({GeometryType.POINT}),
}


_MIN_VERTICES: dict[GeometryType, int] = {
    GeometryType.POINT: 1,
    GeometryType.POLYLINE: 2,
    GeometryType.POLYGON: 3,
    GeometryType.MESH: 3,
}


@dataclass(frozen=True, slots=True)
class FeatureGeometry:
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
    detector: str
    version: str = "1.0"
    inputs_used: frozenset[ContextField] = field(default_factory=frozenset)
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.extra, MappingProxyType):
            object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


@dataclass(frozen=True, slots=True)
class Feature:
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
        self.features = [replace(f, feature_id=i) for i, f in enumerate(self.features, start=1)]

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float] | None:
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
    "ContextField",
    "Feature",
    "FeatureCategory",
    "FeatureCollection",
    "FeatureGeometry",
    "FeatureMetadata",
    "FeatureType",
    "GeometryType",
]
