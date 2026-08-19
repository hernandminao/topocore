"""
Construction/validation tests for SpiralElement.
"""

from __future__ import annotations

import math

import pytest

from topocore.alignment.elements import SpiralElement
from topocore.alignment.exceptions import AlignmentGeometryError
from topocore.geometry.point2d import Point2D


def _valid_entry_spiral(**overrides: object) -> dict:
    """
    A known-consistent entry spiral (radius_start=inf), R=100, L=50.
    ``end`` is the exact clothoid endpoint (independently computed
    via scipy.special.fresnel, same reference value used in
    test_spiral_geometry.py) -- required now that construction
    itself validates chord consistency, not just the pure structural
    checks.
    """
    defaults: dict[str, object] = {
        "start": Point2D(0.0, 0.0),
        "end": Point2D(49.68840292147947, 4.148102426854749),
        "pi": Point2D(33.44311735685825, 0.0),
        "radius_start": math.inf,
        "radius_end": 100.0,
        "length": 50.0,
        "clockwise": False,
    }
    defaults.update(overrides)
    return defaults


def test_spiral_accepts_consistent_geometry() -> None:
    SpiralElement(**_valid_entry_spiral())  # must not raise


def test_spiral_rejects_displaced_pi() -> None:
    """
    PI moved away from the geometrically correct point (but still a
    plausible-looking coordinate, not an obvious outlier) must be
    rejected -- this is the exact "silent data inconsistency" case
    Hernán flagged: a spiral that is otherwise internally consistent
    (start/end/radius/length/clockwise) but whose PI doesn't match.
    """
    with pytest.raises(AlignmentGeometryError, match="pi"):
        SpiralElement(**_valid_entry_spiral(pi=Point2D(40.0, 5.0)))


def test_spiral_rejects_pi_with_wrong_orientation() -> None:
    """
    PI reflected to the wrong side of the initial tangent (same
    distance from the zero-curvature point along the tangent line,
    wrong sign of the offset that would come from a mismatched
    direction) must be rejected.
    """
    with pytest.raises(AlignmentGeometryError, match="pi"):
        SpiralElement(**_valid_entry_spiral(pi=Point2D(-33.44311735685825, 0.0)))


def test_clockwise_spiral_accepts_consistent_pi() -> None:
    """
    Same physical geometry as the CCW reference, mirrored: PI is
    consistent in both directions, not just counterclockwise.
    """
    end_u, end_v = 49.68840292147947, 4.148102426854749
    SpiralElement(
        start=Point2D(0.0, 0.0),
        end=Point2D(end_u, -end_v),
        pi=Point2D(33.44311735685825, 0.0),
        radius_start=math.inf,
        radius_end=100.0,
        length=50.0,
        clockwise=True,
    )  # must not raise


def test_rigid_transform_of_spiral_and_pi_remains_valid() -> None:
    """
    Rotating/translating an entire consistent spiral -- including
    PI -- by the same rigid transform must remain valid: PI
    consistency is a property of the *shape*, not of any particular
    global placement.
    """
    reference = SpiralElement(**_valid_entry_spiral())

    angle = math.radians(52.0)
    offset = Point2D(-300.0, 800.0)

    def rotate_translate(point: Point2D) -> Point2D:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return Point2D(
            offset.x + point.x * cos_a - point.y * sin_a,
            offset.y + point.x * sin_a + point.y * cos_a,
        )

    SpiralElement(
        start=rotate_translate(reference.start),
        end=rotate_translate(reference.end),
        pi=rotate_translate(reference.pi),
        radius_start=reference.radius_start,
        radius_end=reference.radius_end,
        length=reference.length,
        clockwise=reference.clockwise,
    )  # must not raise


def test_spiral_rejects_both_radii_infinite() -> None:
    with pytest.raises(AlignmentGeometryError):
        SpiralElement(**_valid_entry_spiral(radius_start=math.inf, radius_end=math.inf))


def test_spiral_rejects_both_radii_finite() -> None:
    with pytest.raises(AlignmentGeometryError):
        SpiralElement(**_valid_entry_spiral(radius_start=200.0, radius_end=100.0))


def test_spiral_rejects_zero_length() -> None:
    with pytest.raises(AlignmentGeometryError):
        SpiralElement(**_valid_entry_spiral(length=0.0))


def test_spiral_rejects_negative_length() -> None:
    with pytest.raises(AlignmentGeometryError):
        SpiralElement(**_valid_entry_spiral(length=-10.0))


def test_spiral_rejects_infinite_length() -> None:
    with pytest.raises(AlignmentGeometryError):
        SpiralElement(**_valid_entry_spiral(length=math.inf))


def test_spiral_rejects_non_positive_finite_radius() -> None:
    with pytest.raises(AlignmentGeometryError):
        SpiralElement(**_valid_entry_spiral(radius_end=0.0))


def test_spiral_rejects_negative_finite_radius() -> None:
    with pytest.raises(AlignmentGeometryError):
        SpiralElement(**_valid_entry_spiral(radius_end=-100.0))


def test_spiral_rejects_coincident_start_and_end() -> None:
    with pytest.raises(AlignmentGeometryError):
        SpiralElement(**_valid_entry_spiral(end=Point2D(0.0, 0.0)))


def test_spiral_rejects_inconsistent_chord() -> None:
    """
    Correct radius/length but an end point that is NOT where the
    clothoid formula says it should be -- must be rejected at
    construction as geometrically inconsistent, not silently
    accepted (and not deferred to first evaluation).
    """
    with pytest.raises(AlignmentGeometryError, match="inconsistent"):
        SpiralElement(**_valid_entry_spiral(end=Point2D(1000.0, 1000.0)))
