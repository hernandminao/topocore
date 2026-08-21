"""
Regression suite for topocore.features.feature_builder -- PR19.

Includes two real bugs found and fixed in this session:

1. Both _build_legacy and _build_with_grammar only checked
   `len(run) < 2` before constructing line/polygon geometry -- a
   LINE-sized minimum. POLYGON geometry actually requires >= 3
   vertices (FeatureGeometry._MIN_VERTICES). A run/figure of exactly
   2 points for a POLYGON-type code (a plausible field-survey
   scenario -- a missed corner shot) passed this check, then raised
   an UNCAUGHT GeometryError inside FeatureGeometry's own
   validation, crashing the ENTIRE build_features() call for the
   whole survey instead of reporting just that one run as
   INSUFFICIENT_POINTS and skipping it. Confirmed directly in both
   legacy and grammar modes. Fixed by catching GeometryError around
   construction and reporting it the same way as the existing
   INSUFFICIENT_POINTS case.

2. build_features()'s own docstring documents "closure_tolerance
   must be non-negative", but the free function (public API, in
   __all__) never actually validated this -- only FeatureBuilder's
   __init__ did. Calling build_features() directly (not through
   FeatureBuilder) with a negative closure_tolerance silently
   produced different, undocumented behavior (nothing ever closes
   via geometric inference) instead of raising as documented. Fixed
   by adding the same validation to build_features() itself.

Also verifies the full grammar -> builder -> models integration
pipeline end to end: field code -> parser -> assembler -> builder ->
Feature -> FeatureCollection.
"""

from __future__ import annotations

import pytest

from topocore.features.feature_builder import (
    BuildDiagnosticReason,
    FeatureBuilder,
    build_features,
)
from topocore.features.feature_codes import (
    FeatureCodeDefinition,
    FeatureCodeRegistry,
    FeatureGeometryType,
)
from topocore.features.models import FeatureCategory, FeatureType, GeometryType
from topocore.survey.models import SurveyPoint, SurveyPointSet


def _make_registry(*definitions: FeatureCodeDefinition) -> FeatureCodeRegistry:
    registry = FeatureCodeRegistry()
    registry.register_many(definitions)
    return registry


def _polygon_def(code: str = "EDIF") -> FeatureCodeDefinition:
    return FeatureCodeDefinition(
        code=code,
        name="Edificio",
        feature_type=FeatureType.BUILDING,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.POLYGON,
        layer="BUILDINGS",
    )


def _line_def(code: str = "MURO") -> FeatureCodeDefinition:
    return FeatureCodeDefinition(
        code=code,
        name="Muro",
        feature_type=FeatureType.WALL,
        category=FeatureCategory.BUILDING,
        geometry_type=FeatureGeometryType.LINE,
        layer="WALLS",
    )


def _point_def(code: str = "ARBOL") -> FeatureCodeDefinition:
    return FeatureCodeDefinition(
        code=code,
        name="Arbol",
        feature_type=FeatureType.TREE,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POINT,
        layer="TREES",
    )


def _ground_def(code: str = "TERRENO") -> FeatureCodeDefinition:
    return FeatureCodeDefinition(
        code=code,
        name="Terreno",
        feature_type=None,
        category=FeatureCategory.TERRAIN,
        geometry_type=FeatureGeometryType.GROUND,
        layer="TOPO",
    )


def _pt(point_id: str, code: str | None, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SurveyPoint:
    return SurveyPoint(id=point_id, x=x, y=y, z=z, code=code)


# ----------------------------------------------------------------------
# Bug 1: POLYGON with exactly 2 points crashing the whole build.
# ----------------------------------------------------------------------


def test_polygon_with_two_points_reported_not_crashed_legacy() -> None:
    """The exact regression in legacy mode: before the fix, this raised an uncaught GeometryError."""
    registry = _make_registry(_polygon_def())
    points = SurveyPointSet(points=(_pt("p1", "EDIF1", x=0.0, y=0.0), _pt("p2", "EDIF2", x=10.0, y=0.0)))

    result = build_features(points, registry)  # must not raise

    assert len(result.features) == 0
    assert result.diagnostics[0].reason == BuildDiagnosticReason.INSUFFICIENT_POINTS


def test_polygon_with_two_points_reported_not_crashed_grammar() -> None:
    """The exact regression in grammar mode."""
    registry = _make_registry(_polygon_def())
    points = SurveyPointSet(
        points=(
            _pt("p1", "EDIF.1.S", x=0.0, y=0.0),
            _pt("p2", "EDIF.1.X", x=10.0, y=0.0),
        )
    )

    result = build_features(points, registry, use_field_code_grammar=True)  # must not raise

    assert len(result.features) == 0
    assert result.diagnostics[0].reason == BuildDiagnosticReason.INSUFFICIENT_POINTS


def test_polygon_with_three_points_still_builds_correctly() -> None:
    """Confirms the fix didn't break the valid case."""
    registry = _make_registry(_polygon_def())
    points = SurveyPointSet(
        points=(
            _pt("p1", "EDIF1", x=0.0, y=0.0),
            _pt("p2", "EDIF2", x=10.0, y=0.0),
            _pt("p3", "EDIF3", x=10.0, y=10.0),
        )
    )

    result = build_features(points, registry)

    assert len(result.features) == 1
    feature = result.features.features[0]
    assert feature.geometry.geometry_type == GeometryType.POLYGON
    assert feature.geometry.vertex_count == 3


def test_line_with_two_points_still_valid() -> None:
    """LINE genuinely only needs 2 -- confirms the fix's exception-based approach doesn't over-reject."""
    registry = _make_registry(_line_def())
    points = SurveyPointSet(points=(_pt("p1", "MURO1"), _pt("p2", "MURO2", x=10.0)))

    result = build_features(points, registry)

    assert len(result.features) == 1


# ----------------------------------------------------------------------
# Bug 2: build_features() not validating closure_tolerance.
# ----------------------------------------------------------------------


def test_build_features_rejects_negative_closure_tolerance() -> None:
    """The exact regression: calling build_features() directly bypassed FeatureBuilder's own validation."""
    registry = _make_registry(_line_def())
    points = SurveyPointSet(points=(_pt("p1", "MURO1"),))

    with pytest.raises(ValueError, match="non-negative"):
        build_features(points, registry, closure_tolerance=-5.0)


def test_feature_builder_class_still_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FeatureBuilder(closure_tolerance=-1.0)


# ----------------------------------------------------------------------
# Full pipeline integration: field code -> parser -> assembler ->
# builder -> Feature -> FeatureCollection.
# ----------------------------------------------------------------------


def test_full_pipeline_grammar_mode_interleaved_figures() -> None:
    registry = _make_registry(_line_def("MURO"), _line_def("CERCA"))
    points = SurveyPointSet(
        points=(
            _pt("p1", "MURO.1.S", x=0.0, y=0.0),
            _pt("p2", "CERCA.1.S", x=100.0, y=0.0),
            _pt("p3", "MURO.1", x=10.0, y=0.0),
            _pt("p4", "CERCA.1", x=110.0, y=0.0),
            _pt("p5", "MURO.1.E", x=20.0, y=0.0),
            _pt("p6", "CERCA.1.E", x=120.0, y=0.0),
        )
    )

    result = build_features(points, registry, use_field_code_grammar=True)

    assert len(result.features) == 2
    feature_types = {f.feature_type for f in result.features.features}
    assert feature_types == {FeatureType.WALL}
    assert result.diagnostics == ()


def test_ground_points_never_become_features() -> None:
    registry = _make_registry(_ground_def())
    points = SurveyPointSet(points=(_pt("p1", "TERRENO"), _pt("p2", "TERRENO")))

    result = build_features(points, registry)

    assert len(result.features) == 0
    assert len(result.ground) == 2


def test_unregistered_code_reported_as_unmatched() -> None:
    registry = _make_registry()
    points = SurveyPointSet(points=(_pt("p1", "BOGUS"),))

    result = build_features(points, registry)

    assert result.diagnostics[0].reason == BuildDiagnosticReason.UNREGISTERED_CODE
    assert result.unmatched[0].id == "p1"


def test_missing_code_reported() -> None:
    registry = _make_registry()
    points = SurveyPointSet(points=(_pt("p1", None),))

    result = build_features(points, registry)

    assert result.diagnostics[0].reason == BuildDiagnosticReason.MISSING_CODE


def test_feature_ids_normalized_1_to_n() -> None:
    registry = _make_registry(_point_def())
    points = SurveyPointSet(points=(_pt("p1", "ARBOL"), _pt("p2", "ARBOL"), _pt("p3", "ARBOL")))

    result = build_features(points, registry)

    ids = [f.feature_id for f in result.features.features]
    assert ids == [1, 2, 3]


def test_feature_builder_class_matches_free_function() -> None:
    registry = _make_registry(_point_def())
    points = SurveyPointSet(points=(_pt("p1", "ARBOL"),))

    via_class = FeatureBuilder(registry)(points)
    via_function = build_features(points, registry)

    assert len(via_class.features) == len(via_function.features)
