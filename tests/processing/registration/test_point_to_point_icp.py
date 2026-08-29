"""
Coverage audit tests for topocore.processing.registration.point_to_point.PointToPointICP.

_estimate_transformation() is tested directly (bypassing the full
ICPBase.register() loop) for its own documented contract -- its
docstring explicitly states "Raises RegistrationError If there are
fewer than 3 correspondences" as part of ITS OWN interface, not
merely a defensive echo of ICPBase.register()'s own already-tested
`len(correspondences) < 3` check. Matches the same reasoning already
applied to _compute_relative_height() in segmentation/specific.py's
own audit: a private method with a real, directly-testable contract.

name() ("point_to_point_icp") is documented as orphaned -- zero
external callers confirmed via grep, consistent with the same policy
already applied to ICPBase.name()/requires_normals() and other
orphaned metadata methods found throughout this audit. Not tested here.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.processing.exceptions import RegistrationError
from topocore.processing.registration.point_to_point import PointToPointICP

# ----------------------------------------------------------------------
# _estimate_transformation() -- its own documented contract.
# ----------------------------------------------------------------------


def test_fewer_than_three_correspondences_rejected() -> None:
    icp = PointToPointICP()
    correspondences = [(0, 0, 0.1), (1, 1, 0.1)]

    with pytest.raises(RegistrationError, match="At least 3 correspondences"):
        icp._estimate_transformation(correspondences, np.zeros((5, 3)), np.zeros((5, 3)))


def test_svd_failure_on_non_finite_points_rejected() -> None:
    icp = PointToPointICP()
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [np.nan, np.nan, np.nan]])
    target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    correspondences = [(i, i, 0.0) for i in range(4)]

    with pytest.raises(RegistrationError, match="SVD failed"):
        icp._estimate_transformation(correspondences, source, target)


# ----------------------------------------------------------------------
# Happy path -- pure translation, pure rotation, and the reflection-correction branch.
# ----------------------------------------------------------------------


def test_estimates_pure_translation() -> None:
    icp = PointToPointICP()
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = source + np.array([5.0, -2.0, 1.0])
    correspondences = [(i, i, 0.0) for i in range(4)]

    t = icp._estimate_transformation(correspondences, source, target)

    np.testing.assert_allclose(t.translation, [5.0, -2.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(t.rotation, np.eye(3), atol=1e-8)


def test_reflection_correction_branch_produces_proper_rotation() -> None:
    """
    A mirrored (reflected) target configuration is a classic SVD-ICP
    edge case: the naive SVD solution can yield det(rotation) = -1 (an
    improper rotation / reflection), which the algorithm must correct
    back to a proper rotation (det = +1).
    """
    icp = PointToPointICP()
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])  # mirrored in Z
    correspondences = [(i, i, 0.0) for i in range(4)]

    t = icp._estimate_transformation(correspondences, source, target)

    assert np.linalg.det(t.rotation) == pytest.approx(1.0, abs=1e-8)


# ----------------------------------------------------------------------
# _extract_matched_points() -- exercised via _estimate_transformation()'s
# own happy path above; a direct check that ordering is preserved.
# ----------------------------------------------------------------------


def test_extract_matched_points_preserves_correspondence_order() -> None:
    source_points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    target_points = np.array([[1.0, 1.0, 1.0], [11.0, 11.0, 11.0]])
    correspondences = [(2, 1, 0.0), (0, 0, 0.0)]

    matched_source, matched_target = PointToPointICP._extract_matched_points(
        correspondences=correspondences,
        source_points=source_points,
        target_points=target_points,
    )

    np.testing.assert_array_equal(matched_source, [[20.0, 20.0, 20.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(matched_target, [[11.0, 11.0, 11.0], [1.0, 1.0, 1.0]])
