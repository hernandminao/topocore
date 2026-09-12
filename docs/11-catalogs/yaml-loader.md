# YAML Catalog Loader

```python
from topocore.features.catalogs.loaders.yaml_loader import load_yaml

definitions = load_yaml("my_catalog.yaml", encoding="utf-8")
```

**Optional dependency: `PyYAML`.** Imported lazily — TopoCore does
not require YAML support unless this loader is actually used.

```python
>>> load_yaml("my_catalog.yaml")   # PyYAML not installed
Traceback (most recent call last):
    ...
ImportError: YAML catalog support requires PyYAML. Install TopoCore
with the 'yaml' optional dependency.
```

## File shape

Same structure as the JSON loader — a mapping with a `codes`
sequence, parsed with `yaml.safe_load` (never the unsafe loader):

```yaml
codes:
  - code: MIMURO
    name: Custom wall
    geometry_type: line
    feature_type: wall
    category: building
    layer: MY_LAYER
```

Field requirements, types, and defaults are identical to the JSON
loader — see [`json-loader.md`](./json-loader.md)'s field table.

## Structural errors (raised immediately)

```python
>>> load_yaml("list_root.yaml")     # root is not a mapping
TypeError: Catalog root must be a YAML mapping.

>>> load_yaml("no_codes_key.yaml")  # missing or non-sequence "codes"
TypeError: Catalog must contain a 'codes' sequence.
```

## Per-entry errors

As with JSON and CSV, per-entry problems across the whole file are
collected and reported together as a single `ExternalCatalogError` —
see [`loaders.md`](./loaders.md) for the shared mechanism.
