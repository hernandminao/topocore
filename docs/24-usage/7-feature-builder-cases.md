# `FeatureBuilder` — every real case, end to end

## First: do CSV/PTS/XYZ apply to field-code surveys? Confirmed: no — different reader, different pipeline entirely

**`CSVReader`/`PTSReader`/`XYZReader` (documented in
[`point-clouds.md`](./point-clouds.md)) have no concept of a field
code at all** -- confirmed directly, none of their real column
handling recognizes anything like a `code` column. They read plain
spatial coordinates (plus point-cloud-style attributes like
intensity, where the format provides them) into a `PointCloud`, the
exact same type a LAS/LAZ/PLY file produces. A `PointCloud` has no
per-point field code, so `FeatureBuilder` (which requires a
`SurveyPointSet`, not a `PointCloud`) can never be reached from a
CSV/PTS/XYZ file at all.

```python
from topocore.io.ascii.csv.reader import CSVReader

with CSVReader("nube.csv") as r:
    cloud = r.read()
# confirmed real output: reads x,y,z into a PointCloud, no matter
# what other columns the file has -- there is no "code" attribute
# anywhere on the result to check
```

**The "classic" walking convention (whole element, then the next)
never applies to these 3 readers, for a simple reason: the
convention is a rule inside `FeatureBuilder` about grouping points by
field code -- and these 3 readers never produce anything with a field
code to group in the first place.**

```text
SurveyTXTReader (id,x,y,z,code)
        |
        v
   FeatureBuilder   <-- the walking convention (or grammar mode)
        |               matters HERE, because grouping is by code
        v
   Feature (LINE/POLYGON/POINT)


CSVReader / PTSReader / XYZReader (x,y,z only)
        |
        v
   PointCloud        <-- no codes, no "elements" to group at all
        |
        v
   detect_features()  <-- automatic geometric/ML detection, never
                           based on survey walking order
```

**If your CSV/PTS/XYZ data happens to represent linear features**
(e.g. someone captured a road edge as a plain sequence of XYZ
points, with no field code column) -- confirmed: there is no field-
code path for this at all. The only way to turn a `PointCloud` into
`Feature`s is `detect_features()` (automatic geometric/ML detection
-- see [`features.md`](./features.md)), which does not use field
codes or the consecutive-run convention in any way; it never
"discovers" a fence or a road edge from bare, uncoded XYZ points
the way `FeatureBuilder` does from a coded survey. If you need
field-code-style linear features from coordinate data that has no
code column, you have 2 real options: reformat the data into
`id,x,y,z,code` and read it with `SurveyTXTReader` instead
(confirmed, by real execution, this reader does not care about file
extension -- a file named `.csv` with this column structure reads
correctly, see below), or construct `Feature`/`FeatureGeometry`
objects yourself directly (see [`export.md`](./export.md)'s own
worked example).

**Where a field-coded survey uploads from, regardless of its file
extension**: confirmed by real execution -- a file named
`via.csv` containing exactly the same `id,x,y,z,code` content as a
`.txt` file reads identically through `SurveyTXTReader`:

```python
from topocore.survey.reader import SurveyTXTReader

# Confirmed real: extension is irrelevant, only content structure matters
survey_points = SurveyTXTReader("survey_renamed.csv").read()   # works exactly like .txt
```

The choice of reader depends entirely on **what the file's own
content actually is** (coordinates with field codes vs. bare
point-cloud coordinates), never on the file's extension.

## The real-life survey convention that works through `Workflow`

The "classic" convention -- confirmed working, see
[`workflows.md`](./workflows.md) and
[`end-to-end-examples.md`](./end-to-end-examples.md) -- is walking
each linear feature as one uninterrupted run before moving to the
next: the whole centerline, then the whole left edge, then the whole
right edge, never alternating station by station. This is a real,
common field practice (especially with a total station, moving along
one line of sight at a time), distinct from -- but no less real than
-- the cross-section-by-cross-section convention that needs grammar
mode (see [`features.md`](./features.md)).

## Every real `FeatureBuilder` case, confirmed with one real survey file

```text
id,x,y,z,code
1,0.0,0.0,10.0,CERCA            <- LINE, 3-point continuous run
2,10.0,0.0,10.0,CERCA
3,20.0,0.0,10.0,CERCA
4,5.0,5.0,10.5,ARBOL            <- SYMBOL (point), 3 SEPARATE features,
5,15.0,8.0,10.5,ARBOL              never grouped into one line even
6,25.0,3.0,10.5,ARBOL              though consecutive and same code
7,0.0,20.0,9.8,TN                <- GROUND, no feature_type -- feeds
8,10.0,20.0,9.8,TN                   terrain, never becomes a Feature
9,20.0,20.0,9.8,TN
10,30.0,0.0,11.0,BOSQUE           <- POLYGON, 4-point closed run
11,40.0,0.0,11.0,BOSQUE
12,40.0,10.0,11.0,BOSQUE
13,30.0,10.0,11.0,BOSQUE
14,50.0,50.0,10.0,XYZDESCONOCIDO  <- unregistered code
```

```python
from topocore.survey.reader import SurveyTXTReader
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry

survey_points = SurveyTXTReader("todos_los_casos.txt").read()
result = FeatureBuilder(FeatureCodeRegistry.default()).build(survey_points)

for f in result.features:
    print(f.feature_type, f.geometry.geometry_type, len(f.geometry.vertices), f.geometry.closed)
# confirmed real output:
# FeatureType.FENCE   polyline  3  False
# FeatureType.TREE    point     1  False   <- 3 of these, one per ARBOL point
# FeatureType.TREE    point     1  False
# FeatureType.TREE    point     1  False
# FeatureType.FOREST  polygon   4  True

print(result.ground)          # confirmed real output: a 3-tuple of SurveyPoint (the TN points)
print(result.diagnostics)     # confirmed real output:
# (BuildDiagnostic(code='XYZDESCONOCIDO', reason=UNREGISTERED_CODE, point_count=1, point_ids=('14',)),)
print(result.unmatched)       # confirmed real output: the 1 point with the unregistered code
```

**`GROUND`-geometry codes never produce a `Feature`, confirmed by
real execution** -- they're returned separately as `result.ground`
(raw `SurveyPoint`s), meant to feed terrain construction (build a
`TIN` from their coordinates directly, e.g. via
[`primitives.md`](./primitives.md)'s own `TIN.from_points()`), not
`Feature`-based export.

**`SYMBOL`/`POINT`-geometry codes are confirmed never grouped into a
run, even when consecutive and identical** -- each becomes its own
independent, single-vertex `Feature`. Only `LINE`/`POLYGON` geometry
types are subject to the consecutive-run grouping rule at all.

## The 2 grouping modes, side by side, on the same interleaved data

Confirmed by real execution, using the cross-section road survey from
[`end-to-end-examples.md`](./end-to-end-examples.md):

```python
# Legacy mode (default) -- confirmed real output: 0 features, all 15
# points diagnosed INSUFFICIENT_POINTS
FeatureBuilder(FeatureCodeRegistry.default()).build(interleaved_survey_points)

# Grammar mode -- confirmed real output: 3 correct features
# (2 PAVEMENT_EDGE + 1 CENTERLINE, 5 vertices each), even though
# the file interleaves codes station by station
FeatureBuilder(FeatureCodeRegistry.default(), use_field_code_grammar=True).build(interleaved_survey_points)

# Grammar mode on a NON-interleaved (classic) survey -- confirmed
# real output: identical result to legacy mode on the same file.
# Safe to always pass True when the file's own convention is unknown.
FeatureBuilder(FeatureCodeRegistry.default(), use_field_code_grammar=True).build(classic_survey_points)
```

## Where to go next

- [`features.md`](./features.md) -- `FeatureBuilder`'s own full
  constructor, the grammar syntax in detail, and `SideResolver`.
- [`end-to-end-examples.md`](./end-to-end-examples.md) -- the
  complete road-survey pipeline (features, laterality, LandXML
  linearity, export) built on top of these same cases.
- [`point-clouds.md`](./point-clouds.md) -- the 7 point-cloud
  formats (LAS/LAZ/PLY/XYZ/CSV/PTS/E57), for when your data has no
  field codes at all.
