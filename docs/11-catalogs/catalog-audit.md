# Catalog Audit

`topocore.features.catalogs.catalog_audit` is an architectural
validator for a full feature-code catalog — more than statistics,
it's what a contributor runs after adding a batch of new codes for a
company or country-specific catalog, to know immediately whether the
result is still internally consistent.

**Audit ≠ Loader ≠ Registry.** A loader converts one external file
into definitions; a registry stores and exposes definitions at
runtime; the audit checks catalog-wide invariants that only make
sense across the *entire* set of definitions at once — a per-file or
per-registration check can't see them.

## Usage

```python
from topocore.features.catalogs import run_audit, ALL_CODES

report = run_audit(ALL_CODES)
```

`run_audit()` accepts any iterable of `FeatureCodeDefinition` — the
built-in `ALL_CODES`, a custom catalog's raw definitions, or a
combined set — not only a `FeatureCodeRegistry`.

## `CatalogAuditReport`

```python
total_codes: int                    # raw definition count (NOT deduplicated)
total_aliases: int
total_feature_types_used: int
violations: tuple[AuditViolation, ...]
```

**`total_codes` counts raw definitions, not unique ones.** Running
`run_audit(ALL_CODES)` reports `total_codes=159` — the raw count —
while `FeatureCodeRegistry.default()` reports `len(registry) == 158`
after deduplicating the intentional `ARBOL` duplicate (see
[`built-in-catalogs.md`](./built-in-catalogs.md)). Both numbers are
correct; they answer different questions.

Confirmed by direct execution: `run_audit(ALL_CODES)` currently
returns zero violations — the shipped catalog is internally
consistent.

## The 5 checks

1. **Code/alias uniqueness** — defense in depth. `FeatureCodeRegistry`
   already enforces this at registration time, but this check works
   directly on a raw definitions list, without needing a registry.
2. **Non-empty layer** — every definition must declare a layer.
3. **`closed` consistency** — `closed=True` is only meaningful for
   `FeatureGeometryType.POLYGON`.
4. **Geometry validity** — delegated to `catalogs._validation`
   (the same check every loader runs on each entry).
5. **`FeatureType` → `FeatureCategory` consistency** — a given
   `FeatureType` must map to exactly one `FeatureCategory` across the
   *entire* catalog, never varying by which code or catalog file
   declared it.

### Example: a genuine violation

Verified directly — two definitions sharing a `FeatureType` but
disagreeing on `FeatureCategory`:

```python
>>> report = run_audit(conflicting_definitions)
>>> report.violations
(AuditViolation(
    code='tree',
    kind='inconsistent_category',
    message='FeatureType.TREE maps to multiple categories: building, vegetation.'
),)
```

This is exactly the class of mistake that's easy to introduce by
hand across multiple catalog files and easy to miss without an
automated, whole-catalog check.
