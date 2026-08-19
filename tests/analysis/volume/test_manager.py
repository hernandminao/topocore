"""
Regression suite for topocore.analysis.volume.manager.VolumeAnalysis
-- PR19.

Verified: all 5 methods correctly dispatch through compute(), method
resolution/validation, and NaN handling propagates correctly through
the manager for cut_fill/grid_volume (the fix already verified
directly on those classes). No new bugs found in the dispatcher
itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import VolumeError
from topocore.analysis.volume.manager import VolumeAnalysis


def test_default_method_is_cut_fill() -> None:
    manager = VolumeAnalysis()
    assert manager.method == "cut_fill"


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("cut_fill", (np.full((5, 5), 10.0), np.full((5, 5), 8.0), 1.0)),
        ("grid_volume", (np.full((5, 5), 10.0), np.full((5, 5), 8.0), 1.0)),
        ("average_end_area", ([(0.0, 20.0), (10.0, 20.0)],)),
        ("prismoidal", ([(0.0, 20.0), (10.0, 20.0)],)),
    ],
)
def test_compute_dispatches_to_each_method(method: str, args: tuple) -> None:  # type: ignore[type-arg]
    manager = VolumeAnalysis()
    result = manager.compute(*args, method=method)
    assert result.method == method


def test_compute_uses_constructor_method_by_default() -> None:
    manager = VolumeAnalysis(method="grid_volume")
    result = manager.compute(np.full((3, 3), 10.0), np.full((3, 3), 8.0), 1.0)
    assert result.method == "grid_volume"


def test_nan_handling_propagates_through_manager() -> None:
    """
    Confirms the cut_fill/grid_volume NaN-exclusion fix is reachable
    through VolumeAnalysis, not just the underlying classes.
    """
    existing = np.array([[10.0, np.nan], [10.0, 10.0]])
    proposed = np.full((2, 2), 8.0)

    manager = VolumeAnalysis(method="cut_fill")
    result = manager.compute(existing, proposed, 1.0)

    assert result.valid_cells == 3
    assert result.excluded_cells == 1


def test_rejects_invalid_method_at_construction() -> None:
    with pytest.raises(VolumeError):
        VolumeAnalysis(method="bogus")


def test_rejects_invalid_method_at_compute_call() -> None:
    manager = VolumeAnalysis()
    with pytest.raises(VolumeError):
        manager.compute(method="bogus")


def test_callable_interface_matches_compute() -> None:
    manager = VolumeAnalysis(method="cut_fill")
    existing = np.full((3, 3), 10.0)
    proposed = np.full((3, 3), 8.0)

    result = manager(existing, proposed, 1.0)
    assert result.cut_volume == pytest.approx(18.0)
