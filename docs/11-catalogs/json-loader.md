# JSON Catalog Loader

```python
from topocore.features.catalogs.loaders.json_loader import load_json

definitions = load_json("my_catalog.json", encoding="utf-8")
```

No optional dependency — uses Python's standard library `json`
module.

## File shape

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
      "closed": false,
      "aliases": ["POSTE_ESP"]
    }
  ]
}
```

| Field | Required | Type | Notes |
|---|---|---|---|
| `code` | yes | string | primary identifier, case-insensitive at lookup |
| `name` | yes | string | human-readable name |
| `geometry_type` | yes | string | one of `point`, `symbol`, `line`, `polygon`, `ground` |
| `feature_type` | only if `geometry_type != ground` | string | must match an existing `FeatureType` value |
| `category` | yes | string | must match an existing `FeatureCategory` value |
| `layer` | yes | string | non-empty |
| `closed` | no (default `false`) | boolean | only meaningful for `polygon` |
| `aliases` | no (default `[]`) | array of strings | additional codes resolving to the same definition |

## Structural errors (raised immediately, not collected)

A malformed root structure fails before any per-entry validation
runs:

```python
>>> load_json("not_an_object.json")   # root is not a JSON object
TypeError: Catalog root must be a JSON object.

>>> load_json("no_codes_key.json")    # missing or non-list "codes"
TypeError: Catalog must contain a 'codes' array.
```

## Per-entry errors (collected, reported together)

Type errors within an entry (wrong field type) and semantic errors
(unknown enum value, empty required field) are collected across
*every* entry in the file and raised together as a single
`ExternalCatalogError` — see [`loaders.md`](./loaders.md) for the
full mechanism and a verified example with multiple simultaneous
errors.
