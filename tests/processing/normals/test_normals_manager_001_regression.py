"""
Regression suite for NORMALS-MANAGER-001 (fixed in this PR).

Bug: NormalManager.viewpoint's setter validated only `value.shape !=
(3,)`, assuming `value` already had a `.shape` attribute -- a plain
tuple/list raised a confusing AttributeError instead of NormalError.
Additionally (found while fixing the setter): __init__ assigned
`self._viewpoint = viewpoint` DIRECTLY, bypassing the setter (and
thus any validation) entirely -- confirmed directly that
`NormalManager(viewpoint=(1, 2, 3))` was silently accepted at
construction, only to fail later, deep inside _orient_normals(),
with the same confusing AttributeError.

Fix: both the constructor and the setter now use the shared
validate_viewpoint() helper (normals/base.py, already introduced for
PCA-VIEWPOINT-001), giving a consistent NormalError for both paths.

Valid ndarrays and None continue to work exactly as before -- object
identity confirmed unchanged for a valid viewpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.processing.exceptions import NormalError
from topocore.processing.normals.manager import NormalManager


@pytest.mark.parametrize(
    ("bad_viewpoint", "expected_message"),
    [
        ((1.0, 2.0, 3.0), "must be a numpy array"),
        ([1.0, 2.0, 3.0], "must be a numpy array"),
        (np.array([1.0, 2.0]), "must have shape"),
        (np.array(["a", "b", "c"]), "must have a numeric dtype"),
    ],
)
def test_constructor_rejects_invalid_viewpoint_with_normal_error(bad_viewpoint: object, expected_message: str) -> None:
    """Previously a tuple/list was silently accepted at construction, failing later inside _orient_normals()."""
    with pytest.raises(NormalError, match=expected_message):
        NormalManager(viewpoint=bad_viewpoint)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("bad_viewpoint", "expected_message"),
    [
        ((1.0, 2.0, 3.0), "must be a numpy array"),
        ([1.0, 2.0, 3.0], "must be a numpy array"),
        (np.array([1.0, 2.0]), "must have shape"),
        (np.array(["a", "b", "c"]), "must have a numeric dtype"),
    ],
)
def test_setter_rejects_invalid_viewpoint_with_normal_error(bad_viewpoint: object, expected_message: str) -> None:
    """Previously a tuple/list raised a confusing AttributeError deep inside the setter's own shape check."""
    manager = NormalManager()
    with pytest.raises(NormalError, match=expected_message):
        manager.viewpoint = bad_viewpoint  # type: ignore[assignment]


def test_constructor_still_accepts_none_and_valid_ndarray() -> None:
    manager_none = NormalManager(viewpoint=None)
    assert manager_none.viewpoint is None

    valid_viewpoint = np.array([1.0, 2.0, 3.0])
    manager_valid = NormalManager(viewpoint=valid_viewpoint)
    assert manager_valid.viewpoint is valid_viewpoint


def test_setter_still_accepts_none_and_valid_ndarray() -> None:
    manager = NormalManager()

    valid_viewpoint = np.array([1.0, 2.0, 3.0])
    manager.viewpoint = valid_viewpoint
    assert manager.viewpoint is valid_viewpoint

    manager.viewpoint = None
    assert manager.viewpoint is None
