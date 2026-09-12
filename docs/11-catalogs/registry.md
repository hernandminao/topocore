# `FeatureCodeRegistry`

`topocore.features.feature_codes.FeatureCodeRegistry` is the runtime
consumer of every catalog — built-in or custom. It has real,
production consumers: `features/feature_builder.py` and
`workflow/workflow.py` both resolve raw field codes through a
registry during feature construction. This is not speculative,
unused infrastructure.

## Construction

| Constructor | Source |
|---|---|
| `FeatureCodeRegistry()` | empty registry |
| `FeatureCodeRegistry.default()` | the 9 built-in catalogs (`ALL_CODES`), 158 unique definitions |
| `FeatureCodeRegistry.from_json(path)` | a standalone registry from one JSON catalog |
| `FeatureCodeRegistry.from_csv(path)` | a standalone registry from one CSV catalog |
| `FeatureCodeRegistry.from_yaml(path)` | a standalone registry from one YAML catalog (requires `PyYAML`) |

`from_json`/`from_csv`/`from_yaml` build a registry containing
**only** the codes in that file. To layer external codes on top of
the built-in set, use `.default()` followed by `.register_many()` —
see [`custom-catalogs.md`](./custom-catalogs.md).

## Lookup

```python
registry.get(code: str) -> FeatureCodeDefinition | None
```

Lookup is **case-insensitive** and matches either a definition's
primary `code` or any of its `aliases`. Returns `None` for an
unknown code — it does not raise.

```python
>>> registry.get("arbol").code
'ARBOL'
>>> registry.get("NOEXISTE")
None
```

## Registration

```python
registry.register(definition, *, overwrite: bool = False) -> None
registry.register_many(definitions, *, overwrite: bool = False) -> None
```

By default (`overwrite=False`), registering a code or alias that
already resolves to a **different** definition raises `ValueError`.
Re-registering an **identical** definition (same code, same content)
is always accepted silently — this is what makes the built-in
`ARBOL` duplication harmless (see
[`built-in-catalogs.md`](./built-in-catalogs.md)). Pass
`overwrite=True` to force replacement.

## Inspection

```python
len(registry)              # count of unique definitions
list(registry)             # iterate all unique definitions
registry.definitions       # same, as a tuple
```

`.definitions` is deduplicated: if the same `FeatureCodeDefinition`
object ends up reachable via more than one key (its own code plus an
alias, or an intentional duplicate across two catalog files), it is
counted once.

## What a registry does *not* do

A registry does not parse files itself (that's the loaders' job —
see [`loaders.md`](./loaders.md)) and does not check catalog-wide
consistency beyond per-registration collisions (that's
[`catalog_audit`](./catalog-audit.md)'s job).
