"""
topocore.processing.ground.csf
===============================

Cloth Simulation Filter (CSF) ground classification.

This module is the only TopoCore module that knows about the optional
``cloth-simulation-filter`` dependency. The dependency is imported lazily when
classification is requested, so importing TopoCore does not require CSF.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from importlib import import_module
from math import isfinite
from types import ModuleType
from typing import override

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import GroundError
from topocore.processing.types import BoolArray1D

from .base import GroundClassifier, GroundExtractor
from .pmf import _build_ground_cloud_from_mask_streaming


_CSF_INSTALL_ERROR = (
    "CSF ground filtering requires the optional 'cloth-simulation-filter' "
    "package. Install it with: pip install cloth-simulation-filter"
)
_EMPTY_CLOUD_ERROR = "Cannot classify an empty point cloud."
_NO_GROUND_POINTS_ERROR = "No ground points found. Try adjusting CSF parameters."


def _load_csf() -> ModuleType:
    """Import the official CSF Python binding only when it is first used."""
    try:
        return import_module("CSF")
    except (ImportError, OSError) as error:
        raise GroundError(_CSF_INSTALL_ERROR) from error


def _validate_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise GroundError(f"{name} must be positive and finite, got {value}.")


def _validate_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise GroundError(f"{name} must be non-negative and finite, got {value}.")


def _point_count(cloud: PointCloud) -> int:
    count = 0
    for chunk in cloud:
        x = np.asarray(chunk[PointAttribute.X])
        y = np.asarray(chunk[PointAttribute.Y])
        z = np.asarray(chunk[PointAttribute.Z])
        if x.size != y.size or x.size != z.size:
            raise GroundError("Point cloud coordinate attributes must have equal lengths.")
        if x.size and not (
            np.isfinite(x).all()
            and np.isfinite(y).all()
            and np.isfinite(z).all()
        ):
            raise GroundError("Point cloud coordinates must contain only finite values.")
        count += int(x.size)

    if count == 0:
        raise GroundError(_EMPTY_CLOUD_ERROR)
    return count


def _extract_xyz_compact(cloud: PointCloud) -> np.ndarray:
    """
    Copy only XYZ into the contiguous layout required by the CSF binding.

    The cloud is traversed by chunk and no non-coordinate attributes are
    retained. CSF itself requires all input points at once, so one N-by-3
    array is the minimum complete input representation for this adapter.
    """
    count = _point_count(cloud)
    xyz = np.empty((count, 3), dtype=np.float64)
    offset = 0

    for chunk in cloud:
        x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
        y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
        z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)
        size = int(x.size)
        if size == 0:
            continue

        end = offset + size
        xyz[offset:end, 0] = x
        xyz[offset:end, 1] = y
        xyz[offset:end, 2] = z
        offset = end

    return xyz


class CSFGroundClassifier(GroundClassifier):
    """
    Cloth Simulation Filter ground classifier.

    Parameters
    ----------
    cloth_resolution
        Horizontal spacing of cloth particles.
    rigidness
        Cloth rigidity level accepted by the official CSF implementation.
        Valid values are 1, 2 and 3.
    time_step
        Simulation time step.
    class_threshold
        Distance threshold used to classify points against the cloth.
    iterations
        Maximum number of cloth simulation iterations. The upstream binding
        exposes this field with the misspelled name ``interations``; TopoCore
        intentionally provides the correctly spelled public parameter.
    slope_smooth
        Enable post-processing intended for steep slopes.
    """

    __slots__ = (
        "_cloth_resolution",
        "_rigidness",
        "_time_step",
        "_class_threshold",
        "_iterations",
        "_slope_smooth",
    )

    def __init__(
        self,
        cloth_resolution: float = 0.5,
        rigidness: int = 3,
        time_step: float = 0.65,
        class_threshold: float = 0.5,
        iterations: int = 500,
        slope_smooth: bool = False,
    ) -> None:
        _validate_positive("cloth_resolution", cloth_resolution)
        _validate_positive("time_step", time_step)
        _validate_non_negative("class_threshold", class_threshold)

        if (
            isinstance(rigidness, bool)
            or not isinstance(rigidness, (int, np.integer))
            or rigidness not in (1, 2, 3)
        ):
            raise GroundError(f"rigidness must be one of 1, 2 or 3, got {rigidness}.")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, (int, np.integer))
            or iterations < 1
        ):
            raise GroundError(f"iterations must be an integer >= 1, got {iterations}.")
        if not isinstance(slope_smooth, bool):
            raise GroundError(
                f"slope_smooth must be a bool, got {type(slope_smooth).__name__}."
            )

        self._cloth_resolution = float(cloth_resolution)
        self._rigidness = int(rigidness)
        self._time_step = float(time_step)
        self._class_threshold = float(class_threshold)
        self._iterations = int(iterations)
        self._slope_smooth = slope_smooth

    @override
    def classify(self, cloud: PointCloud) -> BoolArray1D:
        """Classify points with the official authors' CSF implementation."""
        csf_module = _load_csf()
        xyz = _extract_xyz_compact(cloud)

        try:
            filter_instance = csf_module.CSF()
            filter_instance.params.cloth_resolution = self._cloth_resolution
            filter_instance.params.rigidness = self._rigidness
            filter_instance.params.time_step = self._time_step
            filter_instance.params.class_threshold = self._class_threshold
            filter_instance.params.interations = self._iterations
            filter_instance.params.bSloopSmooth = self._slope_smooth
            filter_instance.setPointCloud(xyz)

            ground = csf_module.VecInt()
            non_ground = csf_module.VecInt()
            # Avoid the optional cloth_nodes.txt export and its extra I/O.
            filter_instance.do_filtering(ground, non_ground, False)
        except Exception as error:
            raise GroundError(f"CSF ground filtering failed: {error}") from error

        ground_indices = np.fromiter(
            ground,
            dtype=np.intp,
            count=len(ground),
        )
        if ground_indices.size and (
            int(ground_indices.min()) < 0
            or int(ground_indices.max()) >= xyz.shape[0]
        ):
            raise GroundError("CSF returned an out-of-range ground point index.")

        mask = np.zeros(xyz.shape[0], dtype=np.bool_)
        mask[ground_indices] = True
        return mask

    @override
    def name(self) -> str:
        return "csf"


class CSFGroundExtractor(GroundExtractor):
    """Extract ground points using :class:`CSFGroundClassifier`."""

    __slots__ = ("_classifier",)

    def __init__(
        self,
        cloth_resolution: float = 0.5,
        rigidness: int = 3,
        time_step: float = 0.65,
        class_threshold: float = 0.5,
        iterations: int = 500,
        slope_smooth: bool = False,
    ) -> None:
        self._classifier = CSFGroundClassifier(
            cloth_resolution=cloth_resolution,
            rigidness=rigidness,
            time_step=time_step,
            class_threshold=class_threshold,
            iterations=iterations,
            slope_smooth=slope_smooth,
        )

    @override
    def extract(self, cloud: PointCloud) -> PointCloud:
        mask = self._classifier.classify(cloud)
        if not mask.any():
            raise GroundError(_NO_GROUND_POINTS_ERROR)
        return _build_ground_cloud_from_mask_streaming(cloud, mask)

    @override
    def name(self) -> str:
        return "csf"


__all__ = [
    "CSFGroundClassifier",
    "CSFGroundExtractor",
]
