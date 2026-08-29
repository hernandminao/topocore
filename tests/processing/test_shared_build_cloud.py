"""
Coverage audit tests for topocore.processing._shared.build_cloud().

Unlike compute_pca() and flatten_attributes() (where several branches
were confirmed dead/unreachable given prior contracts), build_cloud()
is different: it is a public, general-purpose utility (exported in
__all__) whose `flattened: FlattenedAttributes` parameter is a plain
dict type with no wrapper enforcing it was produced by
flatten_attributes() -- confirmed directly: build_cloud() accepts a
hand-constructed flattened dict just as readily as one produced by
flatten_attributes(). This means _validate_flattened_attributes'
two checks (empty dict, mismatched lengths), when reached via
build_cloud(), protect build_cloud()'s OWN public contract regardless
of what its one current real caller (sampling/random.py, which
always calls flatten_attributes() first) happens to do -- they are
NOT dead code here, unlike the analogous situation for
_validate_cloud_attributes (a private helper with a single,
internally-controlled caller). Every validation branch in
build_cloud() was confirmed reachable via direct execution before
writing this test; none are documented as unreachable.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.processing._shared import FlattenedAttributes, build_cloud
from topocore.processing.exceptions import ProcessingError


@pytest.fixture
def flattened() -> FlattenedAttributes:
    return {
        PointAttribute.X: np.array([10.0, 20.0, 30.0, 40.0]),
        PointAttribute.Y: np.array([1.0, 2.0, 3.0, 4.0]),
        PointAttribute.Z: np.array([0.0, 0.0, 0.0, 0.0]),
    }


# ----------------------------------------------------------------------
# Happy path -- confirmed via direct execution: selection order preserved.
# ----------------------------------------------------------------------


def test_selection_preserves_given_order_not_sorted(
    flattened: FlattenedAttributes,
) -> None:
    result = build_cloud(flattened, np.array([2, 0], dtype=np.intp))

    (chunk,) = list(result)
    np.testing.assert_array_equal(chunk[PointAttribute.X], [30.0, 10.0])
    assert result.point_count == 2


def test_single_chunk_output_structure(flattened: FlattenedAttributes) -> None:
    result = build_cloud(flattened, np.array([0, 1, 2], dtype=np.intp))

    assert result.chunk_count == 1
    assert result.point_count == 3


# ----------------------------------------------------------------------
# Empty selection -- a deliberate, legitimate edge case, not an error.
# ----------------------------------------------------------------------


def test_empty_selection_produces_empty_cloud_without_error(
    flattened: FlattenedAttributes,
) -> None:
    result = build_cloud(flattened, np.array([], dtype=np.intp))

    assert result.point_count == 0


# ----------------------------------------------------------------------
# indices validation.
# ----------------------------------------------------------------------


def test_two_dimensional_indices_rejected(flattened: FlattenedAttributes) -> None:
    with pytest.raises(ProcessingError, match="one-dimensional"):
        build_cloud(flattened, np.array([[0, 1]], dtype=np.intp))


def test_non_integer_indices_rejected(flattened: FlattenedAttributes) -> None:
    with pytest.raises(ProcessingError, match="integer array"):
        build_cloud(flattened, np.array([0.0, 1.0]))


@pytest.mark.parametrize("bad_index", [10, -1])
def test_out_of_range_indices_rejected(flattened: FlattenedAttributes, bad_index: int) -> None:
    with pytest.raises(ProcessingError, match="out-of-range"):
        build_cloud(flattened, np.array([bad_index], dtype=np.intp))


# ----------------------------------------------------------------------
# flattened validation -- reachable here even though it's dead code
# when reached from flatten_attributes() itself (see that audit).
# ----------------------------------------------------------------------


def test_empty_flattened_dict_rejected() -> None:
    with pytest.raises(ProcessingError, match="cannot be empty"):
        build_cloud({}, np.array([], dtype=np.intp))


def test_mismatched_attribute_lengths_rejected() -> None:
    bad_flattened: FlattenedAttributes = {
        PointAttribute.X: np.array([1.0, 2.0, 3.0]),
        PointAttribute.Y: np.array([1.0, 2.0]),
    }

    with pytest.raises(ProcessingError, match="same length"):
        build_cloud(bad_flattened, np.array([0], dtype=np.intp))


def test_accepts_hand_constructed_flattened_not_only_from_flatten_attributes() -> None:
    """
    Confirms build_cloud() is genuinely a general-purpose utility, not
    implicitly coupled to flatten_attributes() as its only valid
    source -- important context for why its validation branches are
    NOT treated as dead code the way flatten_attributes()' own
    private helpers were.
    """
    manual: FlattenedAttributes = {
        PointAttribute.X: np.array([1.0, 2.0, 3.0]),
        PointAttribute.Y: np.array([1.0, 2.0, 3.0]),
        PointAttribute.Z: np.array([1.0, 2.0, 3.0]),
    }

    result = build_cloud(manual, np.array([0, 2], dtype=np.intp))

    assert result.point_count == 2
