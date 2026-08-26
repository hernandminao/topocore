"""
Regression/coverage suite for topocore.geometry.enums.GeometryType
and .protocols -- PR20 coverage phase.

Unlike the other geometry/ files closed in this pass, this pair was
audited with a narrower, contract-focused lens per Hernán's own
instruction: verify what actually exists, don't invent coverage for
its own sake. Two real, notable (not bugs -- nothing in the codebase
consumes either file) findings emerged:

1. topocore.geometry.enums.GeometryType is genuinely UNUSED anywhere
   in the codebase (confirmed via grep across all of topocore/) --
   distinct from, and easily confusable with, the ACTIVELY-used
   topocore.features.models.GeometryType (which has different,
   lowercase string values: "point"/"polygon"/"polyline"/"mesh" vs.
   this one's title-case "Point"/"Polygon"/"Polyline"/etc.). This
   file's 7 members (POINT, SEGMENT, TRIANGLE, POLYGON, POLYLINE,
   BOUNDING_BOX, SURFACE) and topocore.geometry.protocols's 6
   structural protocols (HasArea, HasLength, HasVolume, HasCentroid,
   Bounded, Serializable) both appear to be forward-looking
   scaffolding for a planned geometry-primitives expansion (Segment,
   Triangle, Polygon, Polyline, Surface classes) that doesn't exist
   yet -- only Point2D/Point3D/BBox2D/BBox3D are implemented today,
   and none of them are wired up to satisfy any of these protocols.

2. HasCentroid declares a `centroid` property, but BBox2D/BBox3D
   (the only concrete classes with a conceptually equivalent
   feature) expose `center` instead -- a latent naming mismatch that
   would prevent either class from structurally satisfying
   HasCentroid if that were ever attempted. Not a bug today (nothing
   currently tries), but worth documenting precisely rather than
   silently glossing over.

None of the 6 protocols in protocols.py are @runtime_checkable --
confirmed isinstance() against them raises TypeError, meaning they
can only be used for static (mypy) type checking today, never
runtime structural checks.

No source code changed -- per instruction, only real INCORRECT
BEHAVIOR would warrant a fix, and none was found (only unused,
forward-looking scaffolding with a latent, not-yet-triggered naming
inconsistency).
"""

from __future__ import annotations

import pytest

from topocore.geometry.bbox2d import BBox2D
from topocore.geometry.bbox3d import BBox3D
from topocore.geometry.enums import GeometryType
from topocore.geometry.point2d import Point2D
from topocore.geometry.point3d import Point3D
from topocore.geometry.protocols import (
    Bounded,
    HasArea,
    HasCentroid,
    HasLength,
    HasVolume,
    Serializable,
)

# ----------------------------------------------------------------------
# GeometryType -- all members, StrEnum behavior.
# ----------------------------------------------------------------------


def test_geometry_type_has_exactly_seven_members() -> None:
    assert len(GeometryType) == 7


@pytest.mark.parametrize(
    ("member", "expected_value"),
    [
        (GeometryType.POINT, "Point"),
        (GeometryType.SEGMENT, "Segment"),
        (GeometryType.TRIANGLE, "Triangle"),
        (GeometryType.POLYGON, "Polygon"),
        (GeometryType.POLYLINE, "Polyline"),
        (GeometryType.BOUNDING_BOX, "BoundingBox"),
        (GeometryType.SURFACE, "Surface"),
    ],
)
def test_geometry_type_member_values(member: GeometryType, expected_value: str) -> None:
    assert member.value == expected_value
    assert member == expected_value  # StrEnum: direct string equality
    assert str(member) == expected_value


def test_geometry_type_distinct_from_features_models_geometry_type() -> None:
    """
    Confirms these are genuinely two different enums with different
    conventions -- topocore.features.models.GeometryType (the
    actively-used one) uses lowercase values, this one title-case.
    """
    from topocore.features.models import GeometryType as FeatureGeometryType

    assert GeometryType.POINT.value != FeatureGeometryType.POINT.value
    assert GeometryType.POINT.value == "Point"
    assert FeatureGeometryType.POINT.value == "point"


# ----------------------------------------------------------------------
# protocols.py -- structural definitions, runtime_checkable absence.
# ----------------------------------------------------------------------


def test_protocols_are_not_runtime_checkable() -> None:
    """None of the 6 protocols support isinstance() -- static typing only."""
    box = BBox2D(0.0, 0.0, 10.0, 10.0)
    for protocol in (HasArea, HasLength, HasVolume, HasCentroid, Bounded, Serializable):
        with pytest.raises(TypeError, match="runtime_checkable"):
            isinstance(box, protocol)  # type: ignore[misc]


def test_bbox2d_and_bbox3d_structurally_satisfy_has_area_and_has_volume() -> None:
    """No @runtime_checkable, so this is checked via plain attribute presence, not isinstance()."""
    box2d = BBox2D(0.0, 0.0, 10.0, 10.0)
    box3d = BBox3D(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)

    assert hasattr(box2d, "area")
    assert hasattr(box3d, "volume")


def test_bbox_classes_do_not_structurally_satisfy_has_centroid() -> None:
    """
    The documented naming mismatch: HasCentroid declares `centroid`,
    but BBox2D/BBox3D (and Point2D/Point3D) expose `center` instead.
    """
    box2d = BBox2D(0.0, 0.0, 10.0, 10.0)
    box3d = BBox3D(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    p2d = Point2D(1.0, 1.0)
    p3d = Point3D(1.0, 1.0, 1.0)

    for obj in (box2d, box3d, p2d, p3d):
        assert not hasattr(obj, "centroid")

    assert hasattr(box2d, "center")
    assert hasattr(box3d, "center")


def test_no_current_geometry_class_implements_bounded_or_serializable() -> None:
    """
    Bounded.bounding_box() and Serializable.to_wkt() -- neither
    Point2D/Point3D nor BBox2D/BBox3D implement these today.
    """
    for obj in (Point2D(1.0, 1.0), Point3D(1.0, 1.0, 1.0), BBox2D(0.0, 0.0, 1.0, 1.0)):
        assert not hasattr(obj, "bounding_box")
        assert not hasattr(obj, "to_wkt")


def test_protocols_module_exports() -> None:
    from topocore.geometry import protocols

    assert set(protocols.__all__) == {
        "HasArea",
        "HasLength",
        "HasVolume",
        "HasCentroid",
        "Bounded",
        "Serializable",
    }
