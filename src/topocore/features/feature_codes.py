from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

from topocore.features.models import FeatureCategory, FeatureType, GeometryType


class FeatureGeometryType(Enum):
    POINT = "point"
    SYMBOL = "symbol"
    LINE = "line"
    POLYGON = "polygon"
    GROUND = "ground"


CATALOG_TO_MODEL_GEOMETRY: dict[FeatureGeometryType, GeometryType] = {
    FeatureGeometryType.POINT: GeometryType.POINT,
    FeatureGeometryType.SYMBOL: GeometryType.POINT,
    FeatureGeometryType.LINE: GeometryType.POLYLINE,
    FeatureGeometryType.POLYGON: GeometryType.POLYGON,
    # FeatureGeometryType.GROUND has no GeometryType equivalent --
    # ground-classified points never become a Feature at all.
}
"""
Single source of truth for the catalog-geometry -> model-geometry
mapping, shared by ``catalogs._validation`` and ``feature_builder``
-- previously duplicated as a private copy inside ``_validation.py``.
"""


@dataclass(frozen=True, slots=True)
class FeatureCodeDefinition:
    """
    Parameters
    ----------
    code
        Base survey field code.
    name
        Human-readable name.
    feature_type
        Normalized semantic type -- independent of which catalog or
        field-code convention produced this definition. See
        `topocore.features.models.FeatureType`.

        ``None`` only for ``geometry_type=FeatureGeometryType.GROUND``
        codes (e.g. bare terrain shots): these feed TIN/DTM
        construction directly and never produce a `Feature`, so they
        have no semantic type to declare. For every other
        `geometry_type`, `feature_type` must be set -- enforced by
        `catalogs._validation.validate_definition_geometry`, not by
        this dataclass itself (geometry validation is intentionally
        never an import-time side effect; see `_validation.py`).
    category
        Semantic domain of `feature_type`. Always required, even for
        GROUND codes (e.g. `FeatureCategory.TERRAIN`) -- category
        describes where the code belongs conceptually regardless of
        whether it produces a `Feature`.
    geometry_type
        Construction strategy: how consecutive same-code points
        become geometry. Independent of `feature_type`/`category` --
        see the four-concept separation this enforces (field code /
        feature_type / category / geometry_type / layer, each
        orthogonal).
    layer
        Default CAD/GIS presentation layer.
    closed
        Whether generated geometry should be closed.
    aliases
        Alternative names accepted for the same feature.
    """

    code: str
    name: str
    feature_type: FeatureType | None
    category: FeatureCategory
    geometry_type: FeatureGeometryType
    layer: str
    closed: bool = False
    aliases: tuple[str, ...] = ()


class FeatureCodeRegistry:
    __slots__ = ("_definitions",)

    def __init__(self, definitions: Iterable[FeatureCodeDefinition] | None = None) -> None:
        self._definitions: dict[str, FeatureCodeDefinition] = {}
        if definitions is not None:
            self.register_many(definitions)

    @classmethod
    def default(cls) -> FeatureCodeRegistry:
        """Return the default TopoCore registry, populated from ALL_CODES."""
        from topocore.features.catalogs import ALL_CODES

        registry = cls()
        registry.register_many(ALL_CODES)
        return registry

    @classmethod
    def from_json(cls, path: str) -> FeatureCodeRegistry:
        """
        Build a standalone registry from an external JSON catalog.
        Use `register_many(load_json(path))` on an existing registry
        instead to layer external codes on top of the built-in ones.
        """
        from topocore.features.catalogs.loaders.json_loader import load_json

        return cls(load_json(path))

    @classmethod
    def from_csv(cls, path: str) -> FeatureCodeRegistry:
        """Build a standalone registry from an external CSV catalog."""
        from topocore.features.catalogs.loaders.csv_loader import load_csv

        return cls(load_csv(path))

    @classmethod
    def from_yaml(cls, path: str) -> FeatureCodeRegistry:
        """Build a standalone registry from an external YAML catalog. Requires `pyyaml`."""
        from topocore.features.catalogs.loaders.yaml_loader import load_yaml

        return cls(load_yaml(path))

    def register(self, definition: FeatureCodeDefinition, *, overwrite: bool = False) -> None:
        keys = [
            definition.code.upper(),
            *(alias.upper() for alias in definition.aliases),
        ]

        if not overwrite:
            for key in keys:
                existing = self._definitions.get(key)
                if existing is not None and existing != definition:
                    raise ValueError(
                        f"Code '{key}' is already registered to "
                        f"'{existing.name}' ({existing.code}); refusing to "
                        f"silently overwrite with '{definition.name}' "
                        f"({definition.code}). Pass overwrite=True if this "
                        f"is intentional."
                    )

        for key in keys:
            self._definitions[key] = definition

    def register_many(self, definitions: Iterable[FeatureCodeDefinition], *, overwrite: bool = False) -> None:
        for definition in definitions:
            self.register(definition, overwrite=overwrite)

    def get(self, code: str) -> FeatureCodeDefinition | None:
        return self._definitions.get(code.upper())

    @property
    def definitions(self) -> tuple[FeatureCodeDefinition, ...]:
        return tuple(dict.fromkeys(self._definitions.values()))

    def __len__(self) -> int:
        return len(self.definitions)

    def __iter__(self) -> Iterator[FeatureCodeDefinition]:
        return iter(self.definitions)


__all__ = [
    "CATALOG_TO_MODEL_GEOMETRY",
    "FeatureCodeDefinition",
    "FeatureCodeRegistry",
    "FeatureGeometryType",
]
