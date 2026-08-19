"""
Regression tests for the entry/exit mirroring bug found against a
genuine PLATEIA 2007 export (Sample_Plateia2007LandXML11.XML,
PR18C session). The chord-length check alone did not catch this bug
(hypot(u, v) is insensitive to the sign of v); only comparing 'pi'
against the file's own value exposed it.

Values below are the EXACT real numbers from that file (converted
to Point2D's x=east/y=north convention) -- not synthetic.
"""

from __future__ import annotations

import math

import pytest

from topocore.alignment.elements import SpiralElement
from topocore.geometry.point2d import Point2D


def test_real_entry_spiral_prehodnica_1_accepts_default_tolerance() -> None:
    """
    PREHODNICA 1: entry spiral (radiusStart=INF), rot=cw. This
    element already worked before the entry/exit fix (only exit
    spirals were affected) -- kept as a regression guard.
    """
    SpiralElement(
        start=Point2D(151.428233, 456.863782),
        end=Point2D(242.031321, 450.48292),
        pi=Point2D(212.046751, 455.126629),
        radius_start=math.inf,
        radius_end=363.562418,
        length=90.890605,
        clockwise=True,
        tolerance=2e-6,
    )  # must not raise


def test_real_exit_spiral_prehodnica_2_accepts_default_tolerance() -> None:
    """
    PREHODNICA 2: exit spiral (radiusEnd=INF), rot=cw -- the exact
    element that exposed the entry/exit mirroring bug. Before the
    fix, this raised with 'pi' off by ~5 meters (not explainable by
    rounding); after the fix, it is consistent within the LandXML
    import tolerance.
    """
    SpiralElement(
        start=Point2D(389.730289, 392.584158),
        end=Point2D(460.533107, 335.693068),
        pi=Point2D(414.882772, 375.613856),
        radius_start=363.562418,
        radius_end=math.inf,
        length=90.890605,
        clockwise=True,
        tolerance=2e-6,
    )  # must not raise


def test_real_exit_spiral_rejects_strict_domain_tolerance() -> None:
    """
    Confirms the domain stays strict by default even for this real
    exit spiral -- the ~1e-6 discrepancy from PLATEIA's own 6-decimal
    export precision correctly exceeds the unmodified ~1e-9 domain
    tolerance.
    """
    from topocore.alignment.exceptions import AlignmentGeometryError

    with pytest.raises(AlignmentGeometryError):
        SpiralElement(
            start=Point2D(389.730289, 392.584158),
            end=Point2D(460.533107, 335.693068),
            pi=Point2D(414.882772, 375.613856),
            radius_start=363.562418,
            radius_end=math.inf,
            length=90.890605,
            clockwise=True,
        )


def test_real_exit_spiral_heading_matches_declared_direction() -> None:
    """
    Independent cross-check beyond 'pi' consistency: the tangent
    heading this module computes at the exit spiral's own start
    (via finite difference) must match PLATEIA's own declared
    dirStart for that element, confirming the fix is not merely
    self-consistent but matches the real design software's
    independently-computed direction.
    """
    from topocore.alignment.algorithms.spiral import evaluate_spiral

    spiral = SpiralElement(
        start=Point2D(389.730289, 392.584158),
        end=Point2D(460.533107, 335.693068),
        pi=Point2D(414.882772, 375.613856),
        radius_start=363.562418,
        radius_end=math.inf,
        length=90.890605,
        clockwise=True,
        tolerance=2e-6,
    )

    p0 = evaluate_spiral(spiral, 0.0)
    eps = 1e-3
    p1 = evaluate_spiral(spiral, eps)
    numeric_heading = math.atan2(p1.y - p0.y, p1.x - p0.x) % (2 * math.pi)

    declared_dir_start = 5.689642461808  # PLATEIA's own dirStart attribute

    assert numeric_heading == pytest.approx(declared_dir_start, abs=1e-3)
