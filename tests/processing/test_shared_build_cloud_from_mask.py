"""
Coverage audit tests for topocore.processing._shared.build_cloud_from_mask().

Same status as build_cloud() (see that audit's own module docstring):
a public, general-purpose utility whose `flattened` parameter has no
type-level guarantee of provenance from flatten_attributes(). All 5
real callers (segmentation/base.py, segmentation/specific.py,
filters/manager.py, filters/base.py, classification/base.py) happen
to always call flatten_attributes() immediately before -- confirmed
directly by reading each call site -- but this function's own
validation branches protect its public contract regardless, exactly
as with build_cloud(). Every branch confirmed reachable via direct
execution before writing this test; none are documented as
unreachable.

One additional finding from this audit: build_cloud_from_mask()
delegates to build_cloud() via `np.flatnonzero(mask)`, whose own
output is always a well-formed 1D, intp-dtype, in-range array by
construction -- meaning build_cloud()'s OWN indices-level checks
(ndim, dtype, out-of-range) are unreachable specifically through
this caller, already covered by test_shared_build_cloud.py's own
dedicated tests. Not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.processing._shared import FlattenedAttributes, build_cloud_from_mask
from topocore.processing.exceptions import ProcessingError


@pytest.fixture
def flattened() -> FlattenedAttributes:
    return {
        PointAttribute.X: np.array([10.0, 20.0, 30.0, 40.0]),
        PointAttribute.Y: np.array([1.0, 2.0, 3.0, 4.0]),
        PointAttribute.Z: np.array([0.0, 0.0, 0.0, 0.0]),
    }


# ----------------------------------------------------------------------
# Happy path -- confirmed via direct execution.
# ----------------------------------------------------------------------


def test_mask_selects_correct_points_in_order(flattened: FlattenedAttributes) -> None:
    result = build_cloud_from_mask(flattened, np.array([True, False, True, False]))

    (chunk,) = list(result)
    np.testing.assert_array_equal(chunk[PointAttribute.X], [10.0, 30.0])
    assert result.point_count == 2


# ----------------------------------------------------------------------
# All-False mask -- a deliberate, legitimate edge case (select nothing),
# not an error, matching build_cloud()'s own empty-selection behavior.
# ----------------------------------------------------------------------


def test_all_false_mask_produces_empty_cloud_without_error(
    flattened: FlattenedAttributes,
) -> None:
    result = build_cloud_from_mask(flattened, np.array([False, False, False, False]))

    assert result.point_count == 0


# ----------------------------------------------------------------------
# mask validation.
# ----------------------------------------------------------------------


def test_two_dimensional_mask_rejected(flattened: FlattenedAttributes) -> None:
    with pytest.raises(ProcessingError, match="one-dimensional"):
        build_cloud_from_mask(flattened, np.array([[True, False, True, False]]))


def test_non_boolean_mask_rejected(flattened: FlattenedAttributes) -> None:
    with pytest.raises(ProcessingError, match="boolean array"):
        build_cloud_from_mask(flattened, np.array([1, 0, 1, 0]))


def test_mask_length_mismatch_rejected(flattened: FlattenedAttributes) -> None:
    with pytest.raises(ProcessingError, match="does not match point count"):
        build_cloud_from_mask(flattened, np.array([True, False, True]))


# ----------------------------------------------------------------------
# flattened validation -- reachable here too, same reasoning as build_cloud().
# ----------------------------------------------------------------------


def test_empty_flattened_dict_rejected() -> None:
    with pytest.raises(ProcessingError, match="cannot be empty"):
        build_cloud_from_mask({}, np.array([], dtype=bool))


def test_mismatched_attribute_lengths_rejected() -> None:
    bad_flattened: FlattenedAttributes = {
        PointAttribute.X: np.array([1.0, 2.0, 3.0]),
        PointAttribute.Y: np.array([1.0, 2.0]),
    }

    with pytest.raises(ProcessingError, match="same length"):
        build_cloud_from_mask(bad_flattened, np.array([True, False, True]))
