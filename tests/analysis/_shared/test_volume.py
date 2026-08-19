"""
Regression suite for topocore.analysis._shared.volume -- PR19.

Includes a real bug found and fixed in this session: any grid
containing NaN was rejected outright by validate_volume_arrays().
This made cut/fill volume computation unusable on essentially any
real, irregularly-bounded terrain surface -- confirmed directly with
a genuine DTM produced by topocore.terrain.dtm.DTM.from_tin() (whose
own PR19 fix legitimately leaves cells NaN outside the source TIN's
convex hull, a common, expected shape for real triangulated terrain,
not an error condition). Fixed by excluding NaN cells from the
cut/fill sums (computing only over the area where both surfaces have
valid data) rather than rejecting the whole computation, while still
rejecting genuinely invalid input: infinite values (never a
legitimate NoData marker in this codebase) and all-NaN surfaces
(nothing to compute).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis._shared.volume import compute_cut_fill, validate_volume_arrays
from topocore.analysis.exceptions import VolumeError

# ----------------------------------------------------------------------
# validate_volume_arrays
# ----------------------------------------------------------------------


def test_accepts_partial_nan() -> None:
    existing = np.array([[1.0, np.nan], [2.0, 3.0]])
    proposed = np.array([[0.5, 1.0], [1.5, 2.0]])
    validate_volume_arrays(existing, proposed)  # must not raise


def test_rejects_all_nan_existing() -> None:
    existing = np.full((3, 3), np.nan)
    proposed = np.zeros((3, 3))
    with pytest.raises(VolumeError):
        validate_volume_arrays(existing, proposed)


def test_rejects_all_nan_proposed() -> None:
    existing = np.zeros((3, 3))
    proposed = np.full((3, 3), np.nan)
    with pytest.raises(VolumeError):
        validate_volume_arrays(existing, proposed)


def test_rejects_infinite_existing() -> None:
    existing = np.array([[1.0, np.inf], [2.0, 3.0]])
    proposed = np.zeros((2, 2))
    with pytest.raises(VolumeError):
        validate_volume_arrays(existing, proposed)


def test_rejects_infinite_proposed() -> None:
    existing = np.zeros((2, 2))
    proposed = np.array([[1.0, -np.inf], [2.0, 3.0]])
    with pytest.raises(VolumeError):
        validate_volume_arrays(existing, proposed)


def test_rejects_empty_arrays() -> None:
    with pytest.raises(VolumeError):
        validate_volume_arrays(np.zeros((0, 0)), np.zeros((0, 0)))


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(VolumeError):
        validate_volume_arrays(np.zeros((3, 3)), np.zeros((4, 4)))


def test_rejects_wrong_dimensionality() -> None:
    with pytest.raises(VolumeError):
        validate_volume_arrays(np.zeros(9), np.zeros(9))


# ----------------------------------------------------------------------
# compute_cut_fill -- exact cut/fill convention, verified with an
# unambiguous known case.
# ----------------------------------------------------------------------


def test_pure_cut_known_value() -> None:
    existing = np.full((10, 10), 10.0)
    proposed = np.full((10, 10), 8.0)
    cut, fill, net, valid, excluded = compute_cut_fill(existing, proposed, cell_area=1.0)

    assert cut == pytest.approx(200.0)
    assert fill == pytest.approx(0.0)
    assert net == pytest.approx(200.0)
    assert valid == 100
    assert excluded == 0


def test_pure_fill_known_value() -> None:
    existing = np.full((10, 10), 8.0)
    proposed = np.full((10, 10), 10.0)
    cut, fill, net, _valid, _excluded = compute_cut_fill(existing, proposed, cell_area=1.0)

    assert cut == pytest.approx(0.0)
    assert fill == pytest.approx(200.0)
    assert net == pytest.approx(-200.0)


# ----------------------------------------------------------------------
# NaN exclusion -- the real fix, verified with exact numbers.
# ----------------------------------------------------------------------


def test_nan_cells_excluded_from_sums_not_treated_as_zero() -> None:
    """
    3x3 grid, center row is NaN in `existing` -- must be excluded
    entirely (not contribute 0 to cut, which would be a DIFFERENT,
    wrong result if `existing-proposed` at NaN silently became 0
    instead of skipping the cell).
    """
    existing = np.array(
        [
            [10.0, 10.0, 10.0],
            [np.nan, np.nan, np.nan],
            [10.0, 10.0, 10.0],
        ]
    )
    proposed = np.full((3, 3), 8.0)

    cut, _fill, _net, valid, excluded = compute_cut_fill(existing, proposed, cell_area=1.0)

    assert valid == 6
    assert excluded == 3
    assert cut == pytest.approx(6 * 2.0)  # only the 6 valid cells, 2m cut each


def test_nan_in_either_surface_excludes_the_cell() -> None:
    existing = np.array([[10.0, np.nan], [10.0, 10.0]])
    proposed = np.array([[8.0, 8.0], [np.nan, 8.0]])

    _cut, _fill, _net, valid, excluded = compute_cut_fill(existing, proposed, cell_area=1.0)

    assert valid == 2  # only (0,0) and (1,1) have BOTH surfaces valid
    assert excluded == 2


def test_rejects_nonpositive_cell_area() -> None:
    with pytest.raises(VolumeError):
        compute_cut_fill(np.zeros((2, 2)), np.zeros((2, 2)), cell_area=0.0)


def test_rejects_nonfinite_cell_area() -> None:
    with pytest.raises(VolumeError):
        compute_cut_fill(np.zeros((2, 2)), np.zeros((2, 2)), cell_area=float("inf"))
