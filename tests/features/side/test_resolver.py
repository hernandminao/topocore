"""
Tests for topocore.features.side.resolver.SideResolver.

New capability, PR22-adjacent: laterality (left/right) resolution
for linear features (pavement edges) relative to a reference
CENTERLINE, added as an additive post-processing module over an
already-built FeatureCollection -- no change to features/models.py,
feature_builder.py, catalogs/transportation.py, or the grammar
package.

Every one of the 17 scenarios explicitly enumerated during this
capability's own design review is covered below, each verified with
real geometric construction (not mocked), matching the discipline
used throughout this project's own audit history.

Sign convention: verified numerically (before writing the resolver
itself) that TransversalProfile's own perpendicular-offset direction
(`px, py = -uy, ux`) is mathematically identical to
`cross(tangent, offset) > 0` -- this module's LEFT/RIGHT split reuses
that exact convention, not a second one.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.features.feature_codes import FeatureCodeRegistry
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
from topocore.features.side import Side, SideMethod, SideResolutionError, SideResolver


def _line(
    vertices_xy: list[tuple[float, float]],
    feature_type: FeatureType,
    feature_id: int,
    **kwargs: object,
) -> Feature:
    vertices = np.array([[x, y, 0.0] for x, y in vertices_xy])
    geometry = FeatureGeometry(geometry_type=GeometryType.POLYLINE, vertices=vertices)
    return Feature(
        feature_id=feature_id,
        category=FeatureCategory.INFRASTRUCTURE,
        feature_type=feature_type,
        geometry=geometry,
        **kwargs,  # type: ignore[arg-type]
    )


def _resolve_single_edge(centerline_xy: list[tuple[float, float]], edge_xy: list[tuple[float, float]]) -> Feature:
    collection = FeatureCollection()
    collection.add(_line(centerline_xy, FeatureType.CENTERLINE, 1))
    collection.add(_line(edge_xy, FeatureType.PAVEMENT_EDGE, 2))
    result = SideResolver().resolve(collection)
    return next(f for f in result if f.feature_type == FeatureType.PAVEMENT_EDGE)


# ----------------------------------------------------------------------
# 1-2. Explicit side.
# ----------------------------------------------------------------------


def test_explicit_left_is_used_verbatim_even_when_geometry_disagrees() -> None:
    """Explicit side always wins -- even when the geometric answer would be the opposite."""
    collection = FeatureCollection()
    collection.add(_line([(0, 0), (0, 10)], FeatureType.CENTERLINE, 1))
    collection.add(_line([(5, 0), (5, 10)], FeatureType.PAVEMENT_EDGE, 2, attributes={"side": "LEFT"}))
    result = SideResolver().resolve(collection)
    edge = next(f for f in result if f.feature_type == FeatureType.PAVEMENT_EDGE)
    assert edge.attributes["side"] == Side.LEFT.value
    assert edge.attributes["side_method"] == SideMethod.EXPLICIT.value


def test_explicit_right_case_insensitive() -> None:
    edge = _line([(5, 0), (5, 10)], FeatureType.PAVEMENT_EDGE, 1, attributes={"side": "Right"})
    collection = FeatureCollection()
    collection.add(edge)
    result = SideResolver().resolve(collection)
    resolved = next(iter(result))
    assert resolved.attributes["side"] == Side.RIGHT.value
    assert resolved.attributes["side_method"] == SideMethod.EXPLICIT.value


def test_invalid_explicit_side_raises() -> None:
    edge = _line([(0, 0), (1, 0)], FeatureType.PAVEMENT_EDGE, 1, attributes={"side": "arriba"})
    collection = FeatureCollection()
    collection.add(edge)
    with pytest.raises(SideResolutionError, match="must be 'left' or 'right'"):
        SideResolver().resolve(collection)


# ----------------------------------------------------------------------
# 3. No SIDE, one CENTERLINE.
# ----------------------------------------------------------------------


def test_no_side_with_single_centerline_resolves_geometrically() -> None:
    edge = _resolve_single_edge([(0, 0), (0, 10)], [(-5, 0), (-5, 10)])
    assert edge.attributes["side"] == Side.LEFT.value
    assert edge.attributes["side_method"] == SideMethod.GEOMETRIC.value
    assert edge.attributes["side_reference_feature_id"] == 1


# ----------------------------------------------------------------------
# 4-7. Cardinal orientations.
# ----------------------------------------------------------------------


def test_centerline_oriented_north() -> None:
    left = _resolve_single_edge([(0, 0), (0, 10)], [(-5, 0), (-5, 10)])
    right = _resolve_single_edge([(0, 0), (0, 10)], [(5, 0), (5, 10)])
    assert left.attributes["side"] == Side.LEFT.value
    assert right.attributes["side"] == Side.RIGHT.value


def test_centerline_oriented_south_flips_sides() -> None:
    """The same physical edge position flips LEFT<->RIGHT when the centerline's own direction reverses."""
    south = _resolve_single_edge([(0, 10), (0, 0)], [(-5, 0), (-5, 10)])
    assert south.attributes["side"] == Side.RIGHT.value


def test_centerline_oriented_east() -> None:
    left = _resolve_single_edge([(0, 0), (10, 0)], [(0, 5), (10, 5)])
    right = _resolve_single_edge([(0, 0), (10, 0)], [(0, -5), (10, -5)])
    assert left.attributes["side"] == Side.LEFT.value
    assert right.attributes["side"] == Side.RIGHT.value


def test_centerline_oriented_west_flips_sides() -> None:
    west = _resolve_single_edge([(10, 0), (0, 0)], [(0, 5), (10, 5)])
    assert west.attributes["side"] == Side.RIGHT.value


# ----------------------------------------------------------------------
# 8. Negative coordinates.
# ----------------------------------------------------------------------


def test_negative_coordinates() -> None:
    edge = _resolve_single_edge([(-100, -100), (-100, -90)], [(-105, -100), (-105, -90)])
    assert edge.attributes["side"] == Side.LEFT.value
    assert edge.attributes["side_method"] == SideMethod.GEOMETRIC.value


# ----------------------------------------------------------------------
# 9. Parallel edge (straight, offset).
# ----------------------------------------------------------------------


def test_parallel_edge_resolves_unambiguously() -> None:
    edge = _resolve_single_edge([(0, 0), (0, 50)], [(-3, 0), (-3, 50)])
    assert edge.attributes["side"] == Side.LEFT.value


# ----------------------------------------------------------------------
# 10. Curved edge.
# ----------------------------------------------------------------------


def test_curved_edge_stays_on_one_side() -> None:
    """An edge that wobbles but stays on the same side of the centerline throughout resolves consistently."""
    edge = _resolve_single_edge(
        [(0, 0), (0, 20)],
        [(-5, 0), (-6, 5), (-5, 10), (-6, 15), (-5, 20)],
    )
    assert edge.attributes["side"] == Side.LEFT.value


# ----------------------------------------------------------------------
# 11-12. Multiple CENTERLINEs: unambiguous vs. ambiguous.
# ----------------------------------------------------------------------


def test_multiple_centerlines_clearly_closer_one_is_unambiguous() -> None:
    collection = FeatureCollection()
    collection.add(_line([(0, 0), (0, 10)], FeatureType.CENTERLINE, 1))
    collection.add(_line([(15, 0), (15, 10)], FeatureType.CENTERLINE, 2))
    collection.add(_line([(2, 0), (2, 10)], FeatureType.PAVEMENT_EDGE, 3))
    result = SideResolver().resolve(collection)
    edge = next(f for f in result if f.feature_type == FeatureType.PAVEMENT_EDGE)
    assert edge.attributes["side_method"] == SideMethod.GEOMETRIC.value
    assert edge.attributes["side_reference_feature_id"] == 1


def test_two_comparably_close_centerlines_are_ambiguous() -> None:
    """Parallel roads case: two centerlines within the ambiguity margin of each other -- must not guess."""
    collection = FeatureCollection()
    collection.add(_line([(0, 0), (0, 10)], FeatureType.CENTERLINE, 1))
    collection.add(_line([(8.2, 0), (8.2, 10)], FeatureType.CENTERLINE, 2))
    collection.add(_line([(4, 0), (4, 10)], FeatureType.PAVEMENT_EDGE, 3))
    result = SideResolver().resolve(collection)
    edge = next(f for f in result if f.feature_type == FeatureType.PAVEMENT_EDGE)
    assert edge.attributes["side"] == Side.UNKNOWN.value
    assert edge.attributes["side_method"] == SideMethod.AMBIGUOUS.value


# ----------------------------------------------------------------------
# 13. CENTERLINE too far away.
# ----------------------------------------------------------------------


def test_centerline_beyond_max_distance_is_unknown() -> None:
    edge = _resolve_single_edge([(0, 0), (0, 10)], [(1000, 0), (1000, 10)])
    assert edge.attributes["side"] == Side.UNKNOWN.value
    assert edge.attributes["side_method"] == SideMethod.UNKNOWN.value


# ----------------------------------------------------------------------
# 14. Degenerate geometry.
# ----------------------------------------------------------------------


def test_degenerate_centerline_is_unknown() -> None:
    """A CENTERLINE with a zero-length segment (coincident vertices) has no tangent to offer."""
    edge = _resolve_single_edge([(0, 0), (0, 0)], [(5, 0), (5, 10)])
    assert edge.attributes["side"] == Side.UNKNOWN.value
    assert edge.attributes["side_method"] == SideMethod.UNKNOWN.value


def test_no_centerline_at_all_is_unknown() -> None:
    collection = FeatureCollection()
    collection.add(_line([(5, 0), (5, 10)], FeatureType.PAVEMENT_EDGE, 1))
    result = SideResolver().resolve(collection)
    edge = next(iter(result))
    assert edge.attributes["side"] == Side.UNKNOWN.value
    assert edge.attributes["side_method"] == SideMethod.UNKNOWN.value


def test_edge_point_on_the_centerline_itself_is_ambiguous() -> None:
    """Cross product effectively zero -- the point lies on the centerline's own line, not clearly to either side."""
    edge = _resolve_single_edge([(0, 0), (0, 10)], [(0, 2), (0, 8)])
    assert edge.attributes["side"] == Side.UNKNOWN.value
    assert edge.attributes["side_method"] == SideMethod.AMBIGUOUS.value


# ----------------------------------------------------------------------
# 15. Centerline orientation reversal (covered by south/west tests above,
# restated explicitly for traceability against the design's own list).
# ----------------------------------------------------------------------


def test_reversing_centerline_orientation_inverts_the_answer_for_the_same_edge() -> None:
    forward = _resolve_single_edge([(0, 0), (0, 10)], [(-5, 0), (-5, 10)])
    reversed_ = _resolve_single_edge([(0, 10), (0, 0)], [(-5, 0), (-5, 10)])
    assert forward.attributes["side"] != reversed_.attributes["side"]


# ----------------------------------------------------------------------
# 16-17. Catalog synonyms resolve to the same FeatureType.
# ----------------------------------------------------------------------


def test_centerline_synonyms_all_resolve_to_centerline_type() -> None:
    registry = FeatureCodeRegistry.default()
    for code in ("EJE", "EJEVIAL", "CL"):
        definition = registry.get(code)
        assert definition is not None
        assert definition.feature_type == FeatureType.CENTERLINE


def test_pavement_edge_synonyms_all_resolve_to_pavement_edge_type() -> None:
    registry = FeatureCodeRegistry.default()
    for code in ("BORDE", "BORDEPAV", "PAV", "CALZADA"):
        definition = registry.get(code)
        assert definition is not None
        assert definition.feature_type == FeatureType.PAVEMENT_EDGE


# ----------------------------------------------------------------------
# Additional contract checks beyond the 17 enumerated scenarios.
# ----------------------------------------------------------------------


def test_non_target_feature_types_pass_through_unmodified() -> None:
    collection = FeatureCollection()
    collection.add(_line([(0, 0), (0, 10)], FeatureType.CENTERLINE, 1))
    result = SideResolver().resolve(collection)
    centerline = next(iter(result))
    assert "side" not in centerline.attributes


def test_feature_ids_are_preserved_not_renumbered() -> None:
    collection = FeatureCollection()
    collection.add(_line([(0, 0), (0, 10)], FeatureType.CENTERLINE, 99))
    collection.add(_line([(5, 0), (5, 10)], FeatureType.PAVEMENT_EDGE, 42))
    result = SideResolver().resolve(collection)
    assert [f.feature_id for f in result] == [99, 42]


def test_input_collection_is_not_mutated() -> None:
    collection = FeatureCollection()
    collection.add(_line([(0, 0), (0, 10)], FeatureType.CENTERLINE, 1))
    collection.add(_line([(5, 0), (5, 10)], FeatureType.PAVEMENT_EDGE, 2))
    original_edge = next(f for f in collection if f.feature_type == FeatureType.PAVEMENT_EDGE)

    SideResolver().resolve(collection)

    assert "side" not in original_edge.attributes


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_distance": 0}, "max_distance"),
        ({"max_distance": -1.0}, "max_distance"),
        ({"ambiguity_margin": 1.5}, "ambiguity_margin"),
        ({"ambiguity_margin": -0.1}, "ambiguity_margin"),
        ({"cross_tolerance": -1.0}, "cross_tolerance"),
    ],
)
def test_constructor_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(SideResolutionError, match=match):
        SideResolver(**kwargs)


def test_default_target_types_is_pavement_edge_only() -> None:
    assert SideResolver().target_types == frozenset({FeatureType.PAVEMENT_EDGE})


def test_custom_target_types_can_include_curb() -> None:
    collection = FeatureCollection()
    collection.add(_line([(0, 0), (0, 10)], FeatureType.CENTERLINE, 1))
    collection.add(_line([(5, 0), (5, 10)], FeatureType.CURB, 2))
    resolver = SideResolver(target_types=frozenset({FeatureType.CURB}))
    result = resolver.resolve(collection)
    curb = next(f for f in result if f.feature_type == FeatureType.CURB)
    assert curb.attributes["side"] == Side.RIGHT.value
