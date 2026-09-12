# Catalogs

The feature-code catalog system: `FeatureCodeRegistry` (which
`Workflow.build_features_from_survey()` and `FeatureBuilder` use --
see [`features.md`](./features.md)) plus the machinery to load
additional, external catalogs on top of the 160-code built-in one.

## `FeatureCodeDefinition` — what one catalog code is

```python
from topocore.features.feature_codes import FeatureCodeDefinition, FeatureGeometryType

FeatureCodeDefinition(
    code: str,
    name: str,
    feature_type: FeatureType | None,   # None only for GROUND codes
    category: FeatureCategory,
    geometry_type: FeatureGeometryType,  # POINT | SYMBOL | LINE | POLYGON | GROUND
    layer: str,
    closed: bool = False,
    aliases: tuple[str, ...] = (),
)
```

`geometry_type=GROUND` codes (bare terrain shots) never produce a
`Feature` at all -- they feed TIN/DTM construction directly, and must
declare `feature_type=None`.

## `FeatureCodeRegistry`

```python
from topocore.features.feature_codes import FeatureCodeRegistry

registry = FeatureCodeRegistry.default()   # built-in catalog, 160 codes
registry = FeatureCodeRegistry()            # empty, build up yourself
registry = FeatureCodeRegistry(definitions)  # from any Iterable[FeatureCodeDefinition]

registry.get(code: str) -> FeatureCodeDefinition | None   # case-insensitive
registry.register(definition, *, overwrite: bool = False) -> None
registry.register_many(definitions, *, overwrite: bool = False) -> None
len(registry)
for definition in registry:
    ...
```

`register()` accepts re-registering the exact same definition under
the same code silently (idempotent -- confirmed by identity
comparison, not merely key collision); registering a *different*
definition under an already-used code (or alias) raises `ValueError`
unless `overwrite=True`.

### `register_many()` is not atomic — confirmed real, not a defect

```python
registry = FeatureCodeRegistry.default()   # 159 codes

# 2 new, non-conflicting entries + 1 colliding with an existing code
try:
    registry.register_many(external_definitions)
except ValueError:
    pass

# Confirmed by direct execution: the 2 non-conflicting entries ARE
# now registered, even though the call raised. The pre-existing
# colliding code's own original definition is untouched.
```

No documentation anywhere promises all-or-nothing semantics for this
method, and the partiality is confirmed benign: a pre-existing
definition is never overwritten by a colliding attempt, and the
registry remains fully usable immediately afterward. If you need
all-or-nothing behavior when layering an external catalog, check for
collisions yourself first:

```python
new_codes = [d for d in external_definitions if registry.get(d.code) is None]
registry.register_many(new_codes)
```

## Loading an external catalog

3 formats, each returning `tuple[FeatureCodeDefinition, ...]` --
confirmed genuinely atomic at the loading stage itself: either every
entry in the file parses successfully, or nothing is returned at all
(a single `ExternalCatalogError` collects every problem found, not
just the first).

```python
from topocore.features.catalogs.loaders.json_loader import load_json
from topocore.features.catalogs.loaders.csv_loader import load_csv
from topocore.features.catalogs.loaders.yaml_loader import load_yaml

definitions = load_json(path: str | Path, *, encoding: str = "utf-8")
definitions = load_csv(path: str | Path, *, encoding: str = "utf-8-sig")
definitions = load_yaml(path: str | Path, *, encoding: str = "utf-8")   # requires pyyaml
```

Or build a standalone registry directly from one:

```python
registry = FeatureCodeRegistry.from_json(path)
registry = FeatureCodeRegistry.from_csv(path)
registry = FeatureCodeRegistry.from_yaml(path)
```

### JSON format

```json
{
  "codes": [
    {
      "code": "ARBOL",
      "name": "Tree",
      "geometry_type": "point",
      "feature_type": "tree",
      "category": "vegetation",
      "layer": "TREES",
      "closed": false,
      "aliases": ["TREE", "ARB"]
    }
  ]
}
```

Root must be a JSON object with a `"codes"` array. `closed` must be a
real boolean -- confirmed deliberately strict: a string like
`"false"` is rejected rather than coerced (Python's own `bool("false")`
is `True`, which would silently mean the opposite of what it looks
like).

### YAML format

Same shape as JSON, since both share the same underlying parsing
(`build_raw_entry_from_mapping()`):

```yaml
codes:
  - code: ARBOL
    name: Tree
    geometry_type: point
    feature_type: tree
    category: vegetation
    layer: TREES
    closed: false
    aliases: [TREE, ARB]
```

### CSV format

```text
code,name,geometry_type,feature_type,category,layer,closed,aliases
ARBOL,Tree,point,tree,vegetation,TREES,false,TREE;ARB
PUNTO,Ground shot,ground,,terrain,GROUND,,
```

Required columns: `code, name, geometry_type, feature_type, category,
layer` -- missing any raises `ValueError` naming every missing one at
once, not just the first. `closed` and `aliases` are optional
columns; multiple `aliases` are semicolon-separated. `closed` accepts
`1/true/yes/y/si/sí` or `0/false/no/n` (case-insensitive) -- any
other non-empty value raises rather than guessing; an empty cell
means `False`. A `feature_type` cell may only be left empty for a
`ground`-geometry code (as in the `PUNTO` row above) -- confirmed
directly: an empty `feature_type` for any other `geometry_type`
raises `ExternalCatalogError` ("must declare a feature_type").

## Where to go next

- [`features.md`](./features.md) -- `FeatureType` vs.
  `FeatureCodeDefinition`, and how `FeatureBuilder` resolves (or
  fails to resolve) a survey code into a `Feature`.
- [`workflows.md`](./workflows.md) -- `build_features_from_survey(registry=...)`,
  the `Workflow`-level entry point that consumes a `FeatureCodeRegistry`.
