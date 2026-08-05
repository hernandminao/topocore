from __future__ import annotations

from topocore.dxf.models import LayerStyle
from topocore.features.models import FeatureType

_TERRAIN_COLOR = 30
_BUILDING_COLOR = 1
_INFRASTRUCTURE_COLOR = 8
_DRAINAGE_COLOR = 5
_VEGETATION_COLOR = 3
_UTILITY_COLOR = 6

LAYER_BY_FEATURE_TYPE: dict[FeatureType, str] = {
    FeatureType.BREAKLINE: "TOPO_BREAKLINES",
    FeatureType.EMBANKMENT: "TOPO_EMBANKMENTS",
    FeatureType.SLOPE_CHANGE: "TOPO_SLOPE_CHANGES",
    FeatureType.BUILDING: "BUILDINGS",
    FeatureType.WALL: "WALLS",
    FeatureType.RETAINING_WALL: "RETAINING_WALLS",
    FeatureType.ROOF: "ROOFS",
    FeatureType.ROAD: "ROADS",
    FeatureType.CURB: "CURBS",
    FeatureType.PARKING: "PARKING",
    FeatureType.DRIVEWAY: "DRIVEWAYS",
    FeatureType.DRAINAGE: "DRAINAGE",
    FeatureType.CHANNEL: "CHANNELS",
    FeatureType.MANHOLE: "MANHOLES",
    FeatureType.INSPECTION_CHAMBER: "INSPECTION_CHAMBERS",
    FeatureType.TREE: "TREES",
    FeatureType.SHRUB: "SHRUBS",
    FeatureType.GRASS: "GRASS",
    FeatureType.POLE: "POLES",
    FeatureType.SIGN: "SIGNS",
    FeatureType.LIGHT_POLE: "LIGHT_POLES",
}

CONTOUR_LAYER = "TOPO_CONTOURS"
CONTOUR_LAYER_MINOR = "TOPO_CONTOURS_MINOR"
CONTOUR_LAYER_MAJOR = "TOPO_CONTOURS_MAJOR"


def is_index_contour(
    elevation: float,
    *,
    base: float,
    interval: float,
    every: int,
    tolerance: float = 1e-6,
) -> bool:
    if interval <= 0:
        raise ValueError(f"interval must be positive; got {interval}.")
    if every < 1:
        raise ValueError(f"every must be >= 1; got {every}.")

    step = interval * every
    relative = (elevation - base) / step
    return bool(abs(relative - round(relative)) <= tolerance)


def contour_layer_name(
    *,
    elevation: float | None,
    base: float | None,
    interval: float | None,
    every: int = 5,
) -> str:
    if elevation is None or base is None or interval is None:
        return CONTOUR_LAYER

    return (
        CONTOUR_LAYER_MAJOR
        if is_index_contour(elevation, base=base, interval=interval, every=every)
        else CONTOUR_LAYER_MINOR
    )


def _style(name: str, color: int) -> LayerStyle:
    return LayerStyle(name=name, color=color)


LAYER_STYLES: dict[str, LayerStyle] = {
    **{layer: _style(layer, _TERRAIN_COLOR) for layer in ("TOPO_BREAKLINES", "TOPO_EMBANKMENTS", "TOPO_SLOPE_CHANGES")},
    CONTOUR_LAYER: _style(CONTOUR_LAYER, _TERRAIN_COLOR),
    CONTOUR_LAYER_MINOR: LayerStyle(CONTOUR_LAYER_MINOR, _TERRAIN_COLOR, lineweight=13),
    CONTOUR_LAYER_MAJOR: LayerStyle(CONTOUR_LAYER_MAJOR, _TERRAIN_COLOR, lineweight=30),
    **{layer: _style(layer, _BUILDING_COLOR) for layer in ("BUILDINGS", "WALLS", "RETAINING_WALLS", "ROOFS")},
    **{layer: _style(layer, _INFRASTRUCTURE_COLOR) for layer in ("ROADS", "CURBS", "PARKING", "DRIVEWAYS")},
    **{layer: _style(layer, _DRAINAGE_COLOR) for layer in ("DRAINAGE", "CHANNELS", "MANHOLES", "INSPECTION_CHAMBERS")},
    **{layer: _style(layer, _VEGETATION_COLOR) for layer in ("TREES", "SHRUBS", "GRASS")},
    **{layer: _style(layer, _UTILITY_COLOR) for layer in ("POLES", "SIGNS", "LIGHT_POLES")},
}


def layer_for(feature_type: FeatureType) -> str:
    return LAYER_BY_FEATURE_TYPE[feature_type]


__all__ = [
    "LAYER_BY_FEATURE_TYPE",
    "LAYER_STYLES",
    "CONTOUR_LAYER",
    "CONTOUR_LAYER_MINOR",
    "CONTOUR_LAYER_MAJOR",
    "is_index_contour",
    "contour_layer_name",
    "layer_for",
]
