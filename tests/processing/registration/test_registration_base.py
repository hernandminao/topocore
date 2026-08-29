"""
Coverage audit tests for topocore.processing.registration.base
(Transformation, RegistrationResult, Registrar).

Confirmed via 1 real construction site each (registration/icp.py):
Transformation and RegistrationResult are genuinely active, public
dataclasses with no wrapper preventing malformed construction --
same reasoning as processing._shared.build_cloud()'s own validation.

An unusually large orphan count was found in this module (7 items,
all confirmed via grep across the whole repository, not just
registration/):
  - Transformation.apply() -- confirmed the only `.apply(` hits in
    the codebase belong to an entirely unrelated Filter class
    hierarchy (filters/pass_through.py, etc.); icp.py itself uses
    apply_points() (the raw-array variant), never apply()
    (the PointCloud variant).
  - Transformation.inverse() -- confirmed the one `.inverse(` hit
    found belongs to an unrelated geodesic coordinate transformer,
    not Transformation.
  - Transformation.__matmul__ -- zero usage of the `@` operator
    between Transformations.
  - Transformation.rotation / .translation properties -- zero
    references anywhere in the repository.
  - RegistrationResult.has_source_transformed -- zero callers.
  - Registrar.__call__ -- zero usage of the callable-interface
    pattern, the same orphaned pattern already found for
    NormalManager.__call__, Classifier.__call__, and
    Segmenter.__call__ elsewhere in this audit.
None of these are tested here; they are documented as architectural
debt, not coverage debt.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.processing.exceptions import RegistrationError
from topocore.processing.registration.base import RegistrationResult, Transformation

# ----------------------------------------------------------------------
# Transformation.__post_init__ -- validation.
# ----------------------------------------------------------------------


def test_non_finite_matrix_rejected() -> None:
    with pytest.raises(RegistrationError, match="NaN or Inf"):
        Transformation(np.full((4, 4), np.nan))


def test_wrong_shape_matrix_rejected() -> None:
    with pytest.raises(RegistrationError, match=r"must be \(4, 4\)"):
        Transformation(np.eye(3))


def test_non_orthogonal_rotation_rejected() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = 2.0  # scaling breaks orthogonality
    with pytest.raises(RegistrationError, match="not orthogonal"):
        Transformation(matrix)


def test_reflection_determinant_rejected() -> None:
    matrix = np.diag([1.0, 1.0, -1.0, 1.0])  # orthogonal but det = -1 (a reflection)
    with pytest.raises(RegistrationError, match="determinant \\+1"):
        Transformation(matrix)


def test_wrong_bottom_row_rejected() -> None:
    matrix = np.eye(4)
    matrix[3, 3] = 2.0
    with pytest.raises(RegistrationError, match="bottom row"):
        Transformation(matrix)


# ----------------------------------------------------------------------
# Transformation construction and composition -- confirmed active.
# ----------------------------------------------------------------------


def test_identity_has_zero_rotation_and_translation() -> None:
    t = Transformation.identity()

    np.testing.assert_array_equal(t.rotation, np.eye(3))
    np.testing.assert_array_equal(t.translation, np.zeros(3))


def test_from_rotation_translation_rejects_wrong_rotation_shape() -> None:
    with pytest.raises(RegistrationError, match=r"Rotation must be \(3, 3\)"):
        Transformation.from_rotation_translation(np.eye(2), np.zeros(3))


def test_from_rotation_translation_rejects_wrong_translation_shape() -> None:
    with pytest.raises(RegistrationError, match=r"Translation must be \(3,\)"):
        Transformation.from_rotation_translation(np.eye(3), np.zeros(2))


def test_from_rotation_translation_happy_path() -> None:
    t = Transformation.from_rotation_translation(np.eye(3), np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(t.translation, [1.0, 2.0, 3.0])


def test_compose_combines_translations() -> None:
    t_a = Transformation.from_rotation_translation(np.eye(3), np.array([1.0, 0.0, 0.0]))
    t_b = Transformation.from_rotation_translation(np.eye(3), np.array([0.0, 1.0, 0.0]))

    composed = t_a.compose(t_b)

    np.testing.assert_allclose(composed.translation, [1.0, 1.0, 0.0])


# ----------------------------------------------------------------------
# Transformation.apply_points() -- confirmed active via icp.py.
# ----------------------------------------------------------------------


def test_apply_points_rejects_wrong_shape() -> None:
    t = Transformation.identity()
    with pytest.raises(RegistrationError, match=r"must have shape \(N, 3\)"):
        t.apply_points(np.array([1.0, 2.0, 3.0]))


def test_apply_points_translates_correctly() -> None:
    t = Transformation.from_rotation_translation(np.eye(3), np.array([10.0, 0.0, 0.0]))
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result = t.apply_points(points)

    np.testing.assert_allclose(result, [[11.0, 0.0, 0.0], [10.0, 1.0, 0.0]])


# ----------------------------------------------------------------------
# RegistrationResult.__post_init__ -- validation.
# ----------------------------------------------------------------------


def test_fitness_out_of_range_rejected() -> None:
    t = Transformation.identity()
    with pytest.raises(RegistrationError, match=r"Fitness must be in \[0, 1\]"):
        RegistrationResult(transformation=t, fitness=1.5, rmse=0.1, iterations=5, converged=True)


def test_negative_rmse_rejected() -> None:
    t = Transformation.identity()
    with pytest.raises(RegistrationError, match="RMSE must be non-negative"):
        RegistrationResult(transformation=t, fitness=0.9, rmse=-0.1, iterations=5, converged=True)


def test_negative_iterations_rejected() -> None:
    t = Transformation.identity()
    with pytest.raises(RegistrationError, match="Iterations must be non-negative"):
        RegistrationResult(transformation=t, fitness=0.9, rmse=0.1, iterations=-1, converged=True)


def test_registration_result_happy_path() -> None:
    t = Transformation.identity()
    result = RegistrationResult(transformation=t, fitness=0.9, rmse=0.1, iterations=5, converged=True)

    assert result.fitness == 0.9
    assert result.converged is True
