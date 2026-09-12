# Alignments — `<CoordGeom>` and `<Profile>`

This is the most narrowly-scoped part of TopoCore's LandXML support.
The LandXML schema allows considerably more alignment geometry than
is read here — every exclusion below is deliberate and stated
explicitly in the source, not an oversight.

## Horizontal geometry — `<CoordGeom>`

| Element | Supported | Notes |
|---|---|---|
| `<Line>` | ✅ always | `<Start>`/`<End>`, 2D "north east" points |
| `<Curve crvType="arc">` | ✅ | requires `<Center>` — a chord-defined curve (no `<Center>`) is skipped with a warning |
| `<Curve crvType="...">` (anything else) | ❌ skipped, warned | e.g. three-point/chord-defined curves |
| `<Spiral spiType="clothoid">` | ✅ | one of `radiusStart`/`radiusEnd` must be `"INF"` (a true transition spiral) |
| `<Spiral>` with both radii finite | ❌ skipped, warned | curve-to-curve compound spiral — "not supported in this delivery" |
| `<Spiral spiType="...">` (non-clothoid: cubic, bloss, biquadraticParabola, etc.) | ❌ skipped, warned | `SpiralElement` models a true Euler spiral specifically |

An `<Alignment>` whose `<CoordGeom>` produces zero supported elements
(everything in it was skipped) is itself skipped, with a warning —
not included in the document as an empty alignment.

### Geometric consistency, not just structural parsing

Reading a `<Curve>` or `<Spiral>` doesn't just extract the
attributes — it reconstructs the element and checks that the
declared radius/length is geometrically consistent with the
Start/Center/End/PI coordinates as written. Confirmed directly, this
happens in **two stages**:

```text
Try construction with TopoCore's own strict tolerance (1e-9)
        │
        ├── succeeds ──────────────────────────► use it, no warning
        │
        └── fails
                │
                ▼
        Retry with LANDXML_GEOMETRY_TOLERANCE (2e-6)
                │
                ├── succeeds ─► use it, WARN that domain tolerance
                │               was exceeded (no coordinates changed)
                │
                └── fails ───► LandXMLParseError, naming the
                                alignment and the exact discrepancy
```

`LANDXML_GEOMETRY_TOLERANCE` (2e-6) is not an arbitrary safety
margin — it was derived empirically against two independent
real-world exports (a Civil 3D 2007 file with 47 real `<Curve>`
elements, and a PLATEIA 2007 file with 6 real `<Spiral>` elements),
both showing the same root cause: decimal-place text truncation in
serialized coordinates, scaling with each producer's own export
precision.

Confirmed directly — a `<Curve>` whose `<Center>` is off by
`~1e-6` m (within `LANDXML_GEOMETRY_TOLERANCE` but outside the
domain's own strict default) is accepted with a warning:

```text
Alignment 'CL4': <Curve> accepted within the LandXML import
tolerance (2e-06 m) but exceeds TopoCore's domain tolerance; no
coordinates were modified. Detail: ArcElement.end is
49.999999009999996 from center, expected radius 50.0.
```

A discrepancy beyond even that wider tolerance raises
`LandXMLParseError` rather than silently building an inconsistent
element.

## Vertical geometry — `<Profile>` / `<ProfAlign>`

| Point type | Supported | Becomes |
|---|---|---|
| `<PVI>` | ✅ | a grade-point boundary — contributes a `GradeSegment` to/from its neighbors |
| `<ParaCurve>` with `lengthIn`/`lengthOut` | ✅ | a `VerticalCurve` |
| `<ParaCurve>` with only `length` | ✅ | split evenly: `length_in = length_out = length / 2` |
| Any other PVI variant (e.g. `<CircCurve>`) | ❌ not appended | see below |

An `<Alignment>` with no `<Profile>`/`<ProfAlign>` simply has
`profile=None` — this is the normal, common case (horizontal-only
alignment), not a warning.

**A `<ProfAlign>` with fewer than 2 points is skipped, with a
warning** — there's no meaningful vertical geometry with only one
point.

**A `<ProfAlign>` cannot start or end with a `<ParaCurve>`** — a
vertical curve needs a grade coming in and a grade going out, both
anchored by a `<PVI>`:

```text
Alignment 'CL5': <ProfAlign> cannot start with a <ParaCurve>.
Alignment 'CL5': <ProfAlign> cannot end with a <ParaCurve>.
```

An unsupported point type (e.g. `<CircCurve>`) is simply not
appended to the internal point list — never silently merged into an
incorrect straight segment. Dropping a point can only ever *remove*
information, never fabricate a wrong value; if the omission would
produce an inconsistent profile, `DesignProfile`'s own continuity
check catches it as a validation error rather than accepting a
silently wrong result.

Confirmed directly with a real `<PVI>` → `<ParaCurve>` → `<PVI>`
profile: produces exactly 3 vertical elements
(`GradeSegment`, `VerticalCurve`, `GradeSegment`), matching the
expected station/elevation continuity.

## Writing alignments back out

`LandXMLWriter` writes exactly the horizontal/vertical elements a
`NamedAlignment` already carries — it does not attempt to write any
geometry type the reader itself cannot read. A document built by
reading a LandXML file and passed back to the writer round-trips the
same supported subset; content that was skipped on read (an
unsupported curve, a compound spiral) was never represented in the
in-memory model to begin with, so it cannot reappear on write.
