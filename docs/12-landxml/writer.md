# `LandXMLWriter`

```python
from topocore.io.landxml import LandXMLWriter

writer = LandXMLWriter("output.xml")
report = writer.write(document)
```

## Constructor

```python
LandXMLWriter(path, *, coordinate_precision: int = 10)
```

`coordinate_precision` controls the decimal places written for every
coordinate. The default (10, not the 8 commonly seen in real
Civil3D/TBC exports) is deliberate: `SpiralElement`'s own
chord-consistency check uses an absolute tolerance of `1e-9`
(`topocore.math.config.DEFAULT_MATH_CONFIG`), and 8 decimal places
alone can introduce enough rounding error on a hundreds-magnitude
coordinate to make a perfectly valid spiral fail reconstruction on
re-read. 10 decimal places keeps rounding error safely below that
tolerance.

## `write()`

Validates the document, then serializes it, writing only the
sections that are actually populated:

```python
if document.surfaces:
    self._write_surfaces(...)
if document.point_groups:
    self._write_point_groups(...)
if document.alignments:
    self._write_alignments(...)
```

An entirely empty `LandXMLDocument()` writes successfully — a
minimal, valid, empty LandXML file with just `<Units>` — and can be
read back without error. Confirmed directly.

Returns a `LandXMLWriteReport` (same shape as `LandXMLReadReport` —
counts plus any warnings from the write side).

## Validation before writing

`write()` runs the same `LandXMLValidator.validate_document()` used
after reading, so a document built by hand (not from `read()`) is
checked before anything is written to disk:

```python
>>> writer.write(document_with_two_surfaces_named_the_same)
LandXMLValidationError: Duplicated Surface name(s): ['SAME'].

>>> writer.write(document_with_a_blank_surface_name)
LandXMLValidationError: Blank Surface name is not allowed.
```

Non-finite coordinates in any surface are also rejected before
writing — a `NaN`/`Inf` vertex never reaches the output file.

## Errors

| Exception | When |
|---|---|
| `LandXMLValidationError` | The document violates a semantic invariant (duplicated or blank names, non-finite coordinates) |
| `LandXMLWriteError` | Writing the file to disk fails (e.g. permissions, disk full) |

## Round-trip fidelity

Confirmed directly with a document containing a surface, a point
group, and a line+arc alignment: writing and re-reading preserves
the surface's TIN vertex/triangle counts, the point group's point
count, the alignment's name, and the document's CRS string exactly.
