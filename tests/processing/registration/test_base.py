"""
Regression suite for topocore.processing.registration.base
(Transformation, RegistrationResult) -- PR19.

Verified with a known, deterministic rigid transformation (90-degree
rotation around Z + translation) -- apply/inverse/compose all
confirmed exact. No bugs found.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.processing.exceptions import RegistrationError
from topocore.processing.registration.base import Transformation


@pytest.fixture
def rotation_90z_plus_translation() -> Transformation:
    theta = np.pi / 2
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([1.0, 0.0, 0.0])
    return Transformation.from_rotation_translation(rotation, translation)


def test_apply_points_known_rotation_and_translation(
    rotation_90z_plus_translation: Transformation,
) -> None:
    points = np.array([[1.0, 0.0, 0.0]])
    result = rotation_90z_plus_translation.apply_points(points)
    np.testing.assert_allclose(result, [[1.0, 1.0, 0.0]], atol=1e-9)


def test_inverse_recovers_original_point(
    rotation_90z_plus_translation: Transformation,
) -> None:
    points = np.array([[1.0, 0.0, 0.0]])
    transformed = rotation_90z_plus_translation.apply_points(points)
    recovered = rotation_90z_plus_translation.inverse().apply_points(transformed)
    np.testing.assert_allclose(recovered, points, atol=1e-9)


def test_compose_matches_sequential_application(
    rotation_90z_plus_translation: Transformation,
) -> None:
    second = Transformation.from_rotation_translation(np.eye(3), np.array([5.0, 0.0, 0.0]))
    points = np.array([[1.0, 0.0, 0.0]])

    composed = second.compose(rotation_90z_plus_translation)
    direct = second.apply_points(rotation_90z_plus_translation.apply_points(points))

    np.testing.assert_allclose(composed.apply_points(points), direct, atol=1e-9)


def test_matmul_operator_matches_compose(
    rotation_90z_plus_translation: Transformation,
) -> None:
    second = Transformation.from_rotation_translation(np.eye(3), np.array([5.0, 0.0, 0.0]))
    assert np.allclose(
        (second @ rotation_90z_plus_translation).matrix,
        second.compose(rotation_90z_plus_translation).matrix,
    )


def test_identity_is_a_no_op() -> None:
    identity = Transformation.identity()
    points = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(identity.apply_points(points), points)


def test_rejects_non_orthogonal_rotation() -> None:
    bad_matrix = np.eye(4)
    bad_matrix[:3, :3] = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # not orthogonal
    with pytest.raises(RegistrationError):
        Transformation(bad_matrix)


def test_rejects_improper_rotation_reflection() -> None:
    bad_matrix = np.eye(4)
    bad_matrix[:3, :3] = np.diag([1.0, 1.0, -1.0])  # det = -1, a reflection
    with pytest.raises(RegistrationError):
        Transformation(bad_matrix)


def test_rejects_wrong_shape() -> None:
    with pytest.raises(RegistrationError):
        Transformation(np.eye(3))


def test_rejects_nan_matrix() -> None:
    bad_matrix = np.eye(4)
    bad_matrix[0, 0] = float("nan")
    with pytest.raises(RegistrationError):
        Transformation(bad_matrix)
