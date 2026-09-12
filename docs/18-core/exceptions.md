# `core.exceptions` — 3 exceptions

```python
class TopoCoreError(Exception):
    """Base exception for TopoCore."""

class MathError(TopoCoreError):
    pass

class TopologyError(TopoCoreError):
    pass
```

## `TopoCoreError` — the real foundation of the entire project

Confirmed, by direct inheritance check against every domain
exception module in the codebase: all 12 real exception hierarchies
inherit from `TopoCoreError` directly (`__bases__ == (TopoCoreError,)`),
with zero exceptions and zero inconsistency:

```text
AlignmentError, GPKGError, ProcessingError, FeatureError, GeodesyError,
SurveyError, TerrainError, WorkflowError, PointCloudIOError,
LandXMLError, ASCIIError, DXFError
```

This is the single most heavily-depended-on element in all of
`core`. See [`contracts.md`](./contracts.md) for the full inheritance
account.

## `MathError` — confirmed real, consistent usage

Used across `topocore.math` (`config.py`, `numeric.py`,
`validation.py`) and `topocore.linalg` (`vector2d.py`, `vector3d.py`).
Confirmed with a real, reproducible case: `Point3D`'s own coordinate
validation raises `MathError` directly when constructed with a `NaN`
coordinate -- this is the exact mechanism that made `terrain.Grid.point()`
permanently broken, documented in `16-terrain`'s own audit
(see that block's `validation.md`).

## `TopologyError` — confirmed dead code, not removed

Confirmed by direct search, across the entire real project source:
zero `raise TopologyError` anywhere, zero `except TopologyError`
anywhere, zero import of it anywhere beyond `core/exceptions.py`
itself. Confirmed, separately, by the project's own author: no test
anywhere in the real repository exercises it either -- this is not
"no test visible in this audit's own export," it is a direct
confirmation from the source of truth.

**Not removed during this audit.** Following the same discipline
established during `16-analysis` (a fix was once applied and then
fully reverted after a real, existing test turned out to assert the
"defect" as intended behavior) and reinforced during `16-terrain`
(dead code is documented, not deleted, absent an explicit decision
to do so): `TopologyError` is recorded here as confirmed dead code,
available for a future, deliberate removal decision, not deleted as
part of PR22's own documentation effort. See
[`limitations.md`](./limitations.md).

## Not exported from `core/__init__.py` — confirmed consistent with the whole project's own convention

None of these 3 exceptions (nor anything else in `core`, beyond
`__version__`) is re-exported from `core/__init__.py`. Every real
consumer imports directly from `topocore.core.exceptions` -- confirmed
the actual pattern used by all 12 real domain hierarchies. See
[`overview.md`](./overview.md) for why this is confirmed deliberate,
not a gap: the project's own top-level `topocore/__init__.py` follows
the identical minimal-export pattern.
