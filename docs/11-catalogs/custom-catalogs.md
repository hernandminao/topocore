# Custom Catalogs

A project can define its own feature codes — for a company's own
survey convention, a country-specific standard, or a one-off job —
without modifying TopoCore's source code.

```text
Your file (.json / .csv / .yaml)
              │
              ▼
   JSON / CSV / YAML Loader
              │
              ▼
        parse_entries()
              │
              ├── typed extraction
              ├── enum resolution
              ├── semantic validation
              └── geometry invariant
              │
              ▼
     FeatureCodeDefinition
              │
              ▼
     FeatureCodeRegistry
              │
              ▼
  FeatureBuilder / Workflow
```

## Two ways to bring in a custom catalog

### 1. Standalone registry — only your codes

```python
from topocore.features.feature_codes import FeatureCodeRegistry

registry = FeatureCodeRegistry.from_json("my_catalog.json")
```

Use this when your codes are meant to fully replace the built-in
set (a closed, project-specific convention).

### 2. Layered on top of the built-in catalog

```python
from topocore.features.feature_codes import FeatureCodeRegistry
from topocore.features.catalogs.loaders.json_loader import load_json

registry = FeatureCodeRegistry.default()
registry.register_many(load_json("my_catalog.json"))
```

Use this when you want TopoCore's standard 158 codes available *and*
your own additions on top. This is the more common case in practice.

## What a custom catalog can and cannot do

A custom catalog can define **new codes** with their own name, layer,
geometry, and aliases. It **cannot** invent a new `FeatureType` or
`FeatureCategory` — every code must resolve to one of TopoCore's
existing semantic types. This is deliberate: `FeatureType` is
TopoCore's own ontology, shared across every detector, analysis, and
export path in the engine, and a catalog that could silently add new
values to it would fragment that shared vocabulary.

If your survey convention has a code with no matching `FeatureType`,
the closest existing type is generally the right choice — a wall
detected under a company-specific code is still, semantically, a
`FeatureType.WALL`.

## Verified end-to-end example

The following was verified by direct execution, not merely read from
source.

**`my_catalog.json`**
```json
{
  "codes": [
    {
      "code": "MIPOSTE",
      "name": "Custom utility pole",
      "geometry_type": "point",
      "feature_type": "pole",
      "category": "utility",
      "layer": "MY_LAYER",
      "aliases": ["POSTE_ESP"]
    }
  ]
}
```

```python
>>> from topocore.features.feature_codes import FeatureCodeRegistry
>>> registry = FeatureCodeRegistry.from_json("my_catalog.json")
>>> len(registry)
1
>>> registry.get("MIPOSTE").name
'Custom utility pole'
>>> registry.get("POSTE_ESP") == registry.get("MIPOSTE")
True
```

The entry is accessible by its primary code **and** by any declared
alias — both resolve to the same `FeatureCodeDefinition`.

## What happens with an invalid catalog

A code with an unrecognized `feature_type` is rejected, not silently
guessed at or dropped:

```python
>>> FeatureCodeRegistry.from_json("bad_catalog.json")
Traceback (most recent call last):
    ...
ExternalCatalogError: External feature-code catalog contains 1 error(s):
  [1] X: Unknown feature_type 'tipo_inventado'. External catalogs remap
      field codes onto TopoCore's existing ontology; they do not
      introduce new FeatureType values.
```

See [`loaders.md`](./loaders.md) for the full validation contract,
including how multiple problems in the same file are reported
together instead of one at a time.

## Registering conflicting codes

`FeatureCodeRegistry.register()` (and therefore every `from_*()`
constructor) refuses to silently overwrite a code that already maps
to a *different* definition:

```python
>>> registry = FeatureCodeRegistry.default()
>>> registry.register(some_conflicting_definition)
Traceback (most recent call last):
    ...
ValueError: Code 'ARBOL' is already registered to 'Tree' (ARBOL);
refusing to silently overwrite with 'Something Else' (ARBOL). Pass
overwrite=True if this is intentional.
```

An identical re-registration (same code, same content) is always
accepted silently — see
[`built-in-catalogs.md`](./built-in-catalogs.md#why-159-raw-entries-produce-158-unique-definitions)
for why this matters for the built-in `ARBOL` duplication.

Pass `overwrite=True` if you genuinely intend to replace a built-in
code's definition with your own.
