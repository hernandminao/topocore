# `FeatureCodeDefinition` — the Domain Model

`topocore.features.feature_codes.FeatureCodeDefinition` is the
single data shape every catalog — built-in or custom, JSON, CSV, or
YAML — ultimately produces. Everything else in this section (loaders,
registry, audit) exists to build, store, or verify collections of
this one dataclass.

```python
@dataclass(frozen=True, slots=True)
class FeatureCodeDefinition:
    code: str
    name: str
    feature_type: FeatureType | None
    category: FeatureCategory
    geometry_type: FeatureGeometryType
    layer: str
    closed: bool = False
    aliases: tuple[str, ...] = ()
```

Frozen and slotted: a definition is immutable once constructed, and
equality is by value — this is what makes an intentional exact
duplicate (see `ARBOL` in
[`built-in-catalogs.md`](./built-in-catalogs.md)) harmless rather
than a registration conflict.

## `feature_type` is `None` only for `GROUND` geometry

For every `geometry_type` except `FeatureGeometryType.GROUND`,
`feature_type` is required — enforced by
`catalogs._validation.validate_definition_geometry`, not by the
dataclass constructor itself (geometry validation is intentionally
never an import-time side effect).

`GROUND` codes represent bare terrain shots — they feed TIN/DTM
construction directly and never produce a `Feature` at all, so they
have no semantic type to declare. `category` is still required even
for `GROUND` codes (typically `FeatureCategory.TERRAIN`) — category
describes where a code belongs conceptually regardless of whether it
ever produces a `Feature`.

## `FeatureGeometryType` → model `GeometryType`

A catalog author writes one of 5 catalog-level geometry keywords.
Four of them map onto TopoCore's own `GeometryType` (used by the
`Feature`/`FeatureGeometry` model documented in
[`../07-features/models.md`](../07-features/models.md)); `GROUND`
deliberately has no equivalent, since a `GROUND` point never becomes
a `Feature`.

| Catalog `geometry_type` | Model `GeometryType` |
|---|---|
| `point` | `POINT` |
| `symbol` | `POINT` |
| `line` | `POLYLINE` |
| `polygon` | `POLYGON` |
| `ground` | *(none — never becomes a `Feature`)* |

This mapping (`CATALOG_TO_MODEL_GEOMETRY`) is the single source of
truth shared by `catalogs._validation` and `feature_builder` — it
used to be duplicated as a private copy inside `_validation.py` and
was consolidated to avoid the two copies drifting apart.

## Field reference

See [`json-loader.md`](./json-loader.md)'s field table for the exact
per-field requirements (required vs. optional, expected type, and
defaults) as they apply when authoring a catalog file — the same
table applies identically to JSON, CSV, and YAML catalogs.
