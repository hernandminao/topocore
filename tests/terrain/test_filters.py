"""
Regression suite for topocore.terrain.filters -- PR19.

laplacian_smooth()'s formula verified with a fully-controlled
adjacency graph (a fan of 4 outer points around a center, so the
center's neighbor set is known exactly, not inferred). detect_spikes()
verified both in a normal case (single moderate outlier in a dense
grid -- only that vertex flagged) and its known characteristic with
an extreme, dominant outlier that has very few neighbors (an outlier
that badly skews its own neighbors' local averages will also flag
those neighbors -- an inherent property of neighbor-mean-based
detection, not a bug; confirmed not to happen in realistic denser
configurations).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.exceptions import TerrainError, TerrainValidationError
from topocore.terrain.filters import (
    LaplacianSmoother,
    SpikeDetector,
    detect_spikes,
    laplacian_smooth,
    remove_spikes,
)
from topocore.terrain.tin import TIN


@pytest.fixture
def center_fan_tin() -> TIN:
    """
    Center vertex (index 0) with exactly 4 neighbors (a full fan),
    so its adjacency set is known precisely, not inferred.
    """
    vertices = (
        Point3D(0.0, 0.0, 100.0),  # center -- extreme, deliberate spike
        Point3D(0.0, 1.0, 0.0),
        Point3D(1.0, 0.0, 0.0),
        Point3D(0.0, -1.0, 0.0),
        Point3D(-1.0, 0.0, 0.0),
    )
    simplices = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=np.int32)
    return TIN.from_mesh(vertices, simplices)


# ----------------------------------------------------------------------
# laplacian_smooth: formula verified with known, controlled adjacency.
# ----------------------------------------------------------------------


def test_smooth_factor_1_snaps_exactly_to_neighbor_mean(center_fan_tin: TIN) -> None:
    result = laplacian_smooth(center_fan_tin, iterations=1, factor=1.0)
    assert result.vertices[0].z == pytest.approx(0.0)  # mean of 4 zero neighbors


def test_smooth_factor_half_blends_halfway(center_fan_tin: TIN) -> None:
    result = laplacian_smooth(center_fan_tin, iterations=1, factor=0.5)
    assert result.vertices[0].z == pytest.approx(50.0)  # 100 + 0.5*(0-100)


def test_smooth_preserves_xy_exactly(center_fan_tin: TIN) -> None:
    result = laplacian_smooth(center_fan_tin, iterations=1, factor=1.0)
    for original, smoothed in zip(center_fan_tin.vertices, result.vertices, strict=True):
        assert smoothed.x == original.x
        assert smoothed.y == original.y


def test_smooth_flat_terrain_is_unaffected() -> None:
    vertices = tuple(Point3D(float(i % 3), float(i // 3), 5.0) for i in range(9))
    tin = TIN.from_points(vertices)

    result = laplacian_smooth(tin, iterations=3, factor=1.0)

    for point in result.vertices:
        assert point.z == pytest.approx(5.0)


def test_smooth_does_not_mutate_original_tin(center_fan_tin: TIN) -> None:
    original_z = center_fan_tin.vertices[0].z
    laplacian_smooth(center_fan_tin, iterations=1, factor=1.0)
    assert center_fan_tin.vertices[0].z == original_z


def test_smooth_rejects_zero_iterations(center_fan_tin: TIN) -> None:
    with pytest.raises(TerrainValidationError):
        laplacian_smooth(center_fan_tin, iterations=0, factor=0.5)


@pytest.mark.parametrize("factor", [0.0, -0.1, 1.1])
def test_smooth_rejects_factor_out_of_range(center_fan_tin: TIN, factor: float) -> None:
    with pytest.raises(TerrainValidationError):
        laplacian_smooth(center_fan_tin, iterations=1, factor=factor)


def test_smooth_accepts_boundary_factor_values(center_fan_tin: TIN) -> None:
    laplacian_smooth(center_fan_tin, iterations=1, factor=1.0)  # must not raise


# ----------------------------------------------------------------------
# detect_spikes: normal case (dense grid, moderate outlier).
# ----------------------------------------------------------------------


def _dense_grid_with_one_spike(spike_z: float = 5.0) -> tuple[TIN, int]:
    points = []
    spike_index = None
    for row in range(7):
        for col in range(7):
            z = 0.0
            if row == 3 and col == 3:
                z = spike_z
                spike_index = row * 7 + col
            points.append(Point3D(float(col), float(row), z))
    assert spike_index is not None
    return TIN.from_points(tuple(points)), spike_index


def test_detect_spikes_flags_only_the_true_outlier_in_a_dense_grid() -> None:
    tin, spike_index = _dense_grid_with_one_spike(spike_z=5.0)
    mask = detect_spikes(tin, threshold=1.0)

    flagged = [i for i, flagged in enumerate(mask) if flagged]
    assert flagged == [spike_index]


def test_detect_spikes_flat_terrain_flags_nothing() -> None:
    vertices = tuple(Point3D(float(i % 5), float(i // 5), 10.0) for i in range(25))
    tin = TIN.from_points(vertices)

    mask = detect_spikes(tin, threshold=0.01)
    assert not mask.any()


def test_detect_spikes_extreme_sparse_outlier_also_flags_its_neighbors(center_fan_tin: TIN) -> None:
    """
    Documented characteristic, not a bug: an extreme outlier with
    very few neighbors skews THEIR local average too, so
    neighbor-mean-based detection flags them as well. Confirmed
    (test above) this does not happen with a moderate outlier in a
    denser, more realistic configuration.
    """
    mask = detect_spikes(center_fan_tin, threshold=10.0)
    assert mask.all()  # all 5 vertices flagged in this sparse, extreme case


def test_detect_spikes_rejects_non_positive_threshold(center_fan_tin: TIN) -> None:
    with pytest.raises(TerrainValidationError):
        detect_spikes(center_fan_tin, threshold=0.0)


# ----------------------------------------------------------------------
# remove_spikes
# ----------------------------------------------------------------------


def test_remove_spikes_drops_only_flagged_vertices() -> None:
    tin, spike_index = _dense_grid_with_one_spike(spike_z=5.0)
    result = remove_spikes(tin, threshold=1.0)

    assert result.vertex_count == tin.vertex_count - 1
    assert 5.0 not in [p.z for p in result.vertices]


def test_remove_spikes_raises_when_too_few_vertices_remain(center_fan_tin: TIN) -> None:
    with pytest.raises(TerrainError):
        remove_spikes(center_fan_tin, threshold=10.0)  # flags all 5 -> 0 remain


# ----------------------------------------------------------------------
# LaplacianSmoother / SpikeDetector class wrappers
# ----------------------------------------------------------------------


def test_laplacian_smoother_apply_matches_function(center_fan_tin: TIN) -> None:
    smoother = LaplacianSmoother(center_fan_tin, iterations=1, factor=0.5)
    via_class = smoother.apply()
    via_function = laplacian_smooth(center_fan_tin, iterations=1, factor=0.5)

    assert via_class.vertices[0].z == via_function.vertices[0].z


def test_laplacian_smoother_callable_matches_apply(center_fan_tin: TIN) -> None:
    smoother = LaplacianSmoother(center_fan_tin, iterations=1, factor=0.5)
    assert smoother().vertices[0].z == smoother.apply().vertices[0].z


def test_spike_detector_detect_matches_function(center_fan_tin: TIN) -> None:
    detector = SpikeDetector(center_fan_tin, threshold=10.0)
    np.testing.assert_array_equal(detector.detect(), detect_spikes(center_fan_tin, threshold=10.0))


def test_spike_detector_callable_matches_detect(center_fan_tin: TIN) -> None:
    detector = SpikeDetector(center_fan_tin, threshold=10.0)
    np.testing.assert_array_equal(detector(), detector.detect())


def test_spike_detector_remove_matches_function() -> None:
    tin, _ = _dense_grid_with_one_spike(spike_z=5.0)
    detector = SpikeDetector(tin, threshold=1.0)

    assert detector.remove().vertex_count == remove_spikes(tin, threshold=1.0).vertex_count
