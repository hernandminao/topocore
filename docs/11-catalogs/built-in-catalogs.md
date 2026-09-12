# Built-in Catalogs

TopoCore ships with 9 individually-maintained, individually-tested
catalogs, assembled into a single tuple: `ALL_CODES`.

```python
from topocore.features.catalogs import ALL_CODES
```

## Inventory

| Catalog | Module | Codes |
|---|---|---:|
| `DEFAULT_CODES` | `catalogs.default` | 2 |
| `TERRAIN_CODES` | `catalogs.terrain` | 3 |
| `CONTROL_CODES` | `catalogs.control` | 4 |
| `VEGETATION_CODES` | `catalogs.vegetation` | 22 |
| `STRUCTURE_CODES` | `catalogs.structures` | 29 |
| `TRANSPORTATION_CODES` | `catalogs.transportation` | 23 |
| `DRAINAGE_CODES` | `catalogs.drainage` | 23 |
| `CADASTRE_CODES` | `catalogs.cadastre` | 22 |
| `UTILITY_CODES` | `catalogs.utilities` | 31 |
| **`ALL_CODES` (raw entries)** | `catalogs` (`__init__.py`) | **159** |
| **Unique definitions after registration** | via `FeatureCodeRegistry.default()` | **158** |

Counts confirmed by direct execution against the installed package,
not read off source comments.

## Why 159 raw entries produce 158 unique definitions

This is not data loss. The code `ARBOL` ("Tree") appears twice —
once in `catalogs.default` and once in `catalogs.vegetation` — with
**identical** content (same name, geometry type, layer, feature
type, and category). This duplication is intentional and is
documented directly in `vegetation.py`'s own module docstring.

`FeatureCodeRegistry.register()` explicitly allows re-registering an
identical definition; it only rejects a collision where the **same**
code or alias would resolve to two **different** definitions. Because
both `ARBOL` entries are equal, the registry's own deduplication
(`.definitions` returns a deduplicated tuple) reduces the 159 raw
entries to 158 distinct ones with no error and no information lost.

If you encounter a `159 → 158` count elsewhere, this is the reason —
not a bug.

## `_pending.py` — codes intentionally not yet migrated

`catalogs/_pending.py` is a separate, hand-maintained list of survey
field codes that are known to exist but have **not** been added to
`ALL_CODES` yet. It is not imported into `ALL_CODES` and is not part
of the active catalog.

```text
Built-in catalogs
       │
       └── ALL_CODES
             │
             └── active catalog — consulted at runtime

catalogs/_pending.py
       │
       └── PENDING_CODES
             │
             └── NOT part of ALL_CODES — invisible to any
                 consumer that only inspects the active registry
```

This list exists because a code's *absence* from a catalog can't be
detected automatically — a code that was never migrated is simply
invisible to anything that only inspects `ALL_CODES`. Keeping an
explicit, hand-maintained pending list is what lets
[`catalog_audit.md`](./catalog-audit.md) report "pending" instead of
having no idea a code like `ROCAARBOL` exists at all.

**Practical consequence**: a code that exists only in `_pending.py`
is **not available** to `FeatureCodeRegistry.default()` or to any
consumer resolving codes at runtime. If a survey uses a pending code,
it will not resolve until that code is promoted into one of the 9
active catalogs.

### Example: `ROCAARBOL`

`catalogs/vegetation.py`'s own docstring documents why `ROCAARBOL`
("Tree on Rock") specifically remains pending: its name alone doesn't
disambiguate what it marks — the rock, the tree, or a composite
feature — and inventing a new `FeatureType` for it would misrepresent
the semantics TopoCore is meant to preserve. Three related codes
(`TRONCO`, `TOCON`, `RAIZ` — trunk, stump, root) were previously
deferred alongside it but were promoted once each was given an
unambiguous, specific `FeatureType` (`TREE_TRUNK`, `TREE_STUMP`,
`TREE_ROOT`).
