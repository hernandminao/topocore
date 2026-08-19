"""
Tests for the optional ``tolerance`` constructor parameter on
ArcElement/SpiralElement, added specifically to support
topocore.io.landxml.LandXMLReader importing real-world engineering
data.

The ArcElement case below uses the EXACT values from a genuine
Autodesk Civil 3D 2007 export (GSG_features_alignments.xml,
Alignment "Conway Farms Drive", first <Curve>) -- not synthetic
data. See the PR18C session notes for the full audit (47 real
curves, 38 exceeding the domain's default tolerance, max observed
deviation 7.995e-9).
"""

from __future__ import annotations

import pytest

from topocore.alignment.elements import ArcElement
from topocore.alignment.exceptions import AlignmentGeometryError
from topocore.geometry.point2d import Point2D

# Real Civil 3D 2007 export values -- Alignment "Conway Farms Drive".
_REAL_START = Point2D(21201.48866028, 20216.84509917)
_REAL_CENTER = Point2D(21200.39648707, 20536.84323737)
_REAL_END = Point2D(21178.04077632, 20217.62509332)
_REAL_RADIUS = 320.00000202


def test_default_tolerance_rejects_real_civil3d_curve() -> None:
    """
    Confirms the domain stays strict by default -- this real curve's
    Start/Center/End are ~1.5e-9 off from the declared radius, which
    the default (unchanged) tolerance correctly rejects.
    """
    with pytest.raises(AlignmentGeometryError):
        ArcElement(
            start=_REAL_START,
            end=_REAL_END,
            center=_REAL_CENTER,
            radius=_REAL_RADIUS,
            clockwise=False,
        )


def test_explicit_tolerance_accepts_real_civil3d_curve() -> None:
    ArcElement(
        start=_REAL_START,
        end=_REAL_END,
        center=_REAL_CENTER,
        radius=_REAL_RADIUS,
        clockwise=False,
        tolerance=1e-8,
    )  # must not raise


def test_tolerance_too_narrow_still_rejects() -> None:
    with pytest.raises(AlignmentGeometryError):
        ArcElement(
            start=_REAL_START,
            end=_REAL_END,
            center=_REAL_CENTER,
            radius=_REAL_RADIUS,
            clockwise=False,
            tolerance=1e-10,  # narrower than default -- still rejects
        )


def test_tolerance_is_not_stored_as_instance_state() -> None:
    """
    tolerance is a construction-time-only InitVar -- it must not
    linger as part of the object's persisted state/slots.
    """
    element = ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, 50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=False,
        tolerance=1e-6,
    )

    assert "tolerance" not in ArcElement.__slots__
    assert "tolerance" not in repr(element)


def test_grossly_inconsistent_geometry_still_rejected_even_with_wide_tolerance() -> None:
    """
    A wide tolerance is not a blank check -- geometry that is
    genuinely wrong (not just rounded) must still be rejected.
    """
    with pytest.raises(AlignmentGeometryError):
        ArcElement(
            start=Point2D(50.0, 0.0),
            end=Point2D(0.0, 50.0),
            center=Point2D(0.0, 0.0),
            radius=1000.0,  # wildly wrong, not a rounding artifact
            clockwise=False,
            tolerance=1e-8,
        )
