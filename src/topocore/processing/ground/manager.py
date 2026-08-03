"""
topocore.processing.ground.manager
==================================

Ground classification manager with automatic method selection.

This module provides a high-level manager that selects the appropriate
ground classification method based on the point cloud characteristics
and user preferences.

The manager supports:
- Automatic method selection
- Method switching
- Caching of computed ground masks

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.cache import LRUCache
from topocore.processing.exceptions import GroundError
from topocore.processing.types import BoolArray1D, FloatArray1D

from .base import GroundClassifier, GroundElevationEstimator, GroundExtractor
from .csf import CSFGroundClassifier, CSFGroundExtractor
from .grid import (
    GridGroundClassifier,
    GridGroundElevationEstimator,
    GridGroundExtractor,
    _build_ground_cloud_from_mask,
    _extract_xyz,
)
from .grid_adaptive import AdaptiveGridGroundClassifier, AdaptiveGridGroundExtractor
from .pmf import PMFGroundClassifier, PMFGroundExtractor
from .progressive_tin import ProgressiveTINGroundClassifier, ProgressiveTINGroundExtractor

_NO_GROUND_POINTS_ERROR = "No ground points found."
_UNSUPPORTED_METHOD_ERROR = "Unsupported method: {method}. Supported: {supported}"


class GroundManager:
    """
    High-level manager for ground classification and extraction.

    This class provides a unified interface for ground classification
    with automatic method selection and caching.

    Examples
    --------
    >>> manager = GroundManager(method="grid", cell_size=1.0)
    >>> mask = manager.classify(cloud)
    >>> ground_cloud = manager.extract(cloud)

    >>> manager.method = "progressive_tin"
    >>> mask = manager.classify(cloud)
    """

    __slots__ = (
        "_method",
        "_cell_size",
        "_height_threshold",
        "_max_distance",
        "_max_angle",
        "_max_iterations",
        "_min_cell_size",
        "_max_cell_size",
        "_slope_threshold",
        "_use_multiresolution",
        "_pmf_initial_distance",
        "_pmf_max_distance",
        "_pmf_slope",
        "_pmf_max_window_size",
        "_pmf_window_base",
        "_pmf_exponential",
        "_pmf_max_grid_cells",
        "_csf_cloth_resolution",
        "_csf_rigidness",
        "_csf_time_step",
        "_csf_class_threshold",
        "_csf_iterations",
        "_csf_slope_smooth",
        "_cache",
        "_cloud_id",
    )

    _SUPPORTED_METHODS: ClassVar[dict[str, type[GroundClassifier]]] = {
        "grid": GridGroundClassifier,
        "adaptive_grid": AdaptiveGridGroundClassifier,
        "progressive_tin": ProgressiveTINGroundClassifier,
        "pmf": PMFGroundClassifier,
        "csf": CSFGroundClassifier,
    }

    _EXTRACTORS: ClassVar[dict[str, type[GroundExtractor]]] = {
        "grid": GridGroundExtractor,
        "adaptive_grid": AdaptiveGridGroundExtractor,
        "progressive_tin": ProgressiveTINGroundExtractor,
        "pmf": PMFGroundExtractor,
        "csf": CSFGroundExtractor,
    }

    _ELEVATION_ESTIMATORS: ClassVar[dict[str, type[GroundElevationEstimator] | None]] = {
        "grid": GridGroundElevationEstimator,
        "adaptive_grid": None,
        "progressive_tin": None,
        "pmf": None,
        "csf": None,
    }

    def __init__(
        self,
        method: str = "grid",
        cell_size: float = 1.0,
        height_threshold: float = 0.2,
        max_distance: float = 0.5,
        max_angle: float = 15.0,
        max_iterations: int = 10,
        cache_size: int = 16,
        min_cell_size: float = 0.5,
        max_cell_size: float = 5.0,
        slope_threshold: float = 30.0,
        use_multiresolution: bool = True,
        pmf_initial_distance: float = 0.15,
        pmf_max_distance: float = 2.5,
        pmf_slope: float = 1.0,
        pmf_max_window_size: int = 33,
        pmf_window_base: float = 2.0,
        pmf_exponential: bool = True,
        pmf_max_grid_cells: int = 8_000_000,
        csf_cloth_resolution: float = 0.5,
        csf_rigidness: int = 3,
        csf_time_step: float = 0.65,
        csf_class_threshold: float = 0.5,
        csf_iterations: int = 500,
        csf_slope_smooth: bool = False,
    ) -> None:
        if method not in self._SUPPORTED_METHODS:
            raise GroundError(
                _UNSUPPORTED_METHOD_ERROR.format(
                    method=method,
                    supported=list(self._SUPPORTED_METHODS.keys()),
                )
            )

        self._method = method
        self._cell_size = cell_size
        self._height_threshold = height_threshold
        self._max_distance = max_distance
        self._max_angle = max_angle
        self._max_iterations = max_iterations
        self._min_cell_size = min_cell_size
        self._max_cell_size = max_cell_size
        self._slope_threshold = slope_threshold
        self._use_multiresolution = use_multiresolution
        self._pmf_initial_distance = pmf_initial_distance
        self._pmf_max_distance = pmf_max_distance
        self._pmf_slope = pmf_slope
        self._pmf_max_window_size = pmf_max_window_size
        self._pmf_window_base = pmf_window_base
        self._pmf_exponential = pmf_exponential
        self._pmf_max_grid_cells = pmf_max_grid_cells
        self._csf_cloth_resolution = csf_cloth_resolution
        self._csf_rigidness = csf_rigidness
        self._csf_time_step = csf_time_step
        self._csf_class_threshold = csf_class_threshold
        self._csf_iterations = csf_iterations
        self._csf_slope_smooth = csf_slope_smooth
        self._cache: LRUCache[tuple[str, int], BoolArray1D] = LRUCache(maxsize=cache_size)
        self._cloud_id = 0

    @property
    def method(self) -> str:
        """Get the current method."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """Set the method and invalidate cached results."""
        if value not in self._SUPPORTED_METHODS:
            raise GroundError(
                _UNSUPPORTED_METHOD_ERROR.format(
                    method=value,
                    supported=list(self._SUPPORTED_METHODS.keys()),
                )
            )
        self._method = value
        self._cache.clear()

    @property
    def cell_size(self) -> float:
        return self._cell_size

    @cell_size.setter
    def cell_size(self, value: float) -> None:
        if value <= 0:
            raise GroundError(f"cell_size must be positive, got {value}.")
        self._cell_size = value
        self._cache.clear()

    @property
    def height_threshold(self) -> float:
        return self._height_threshold

    @height_threshold.setter
    def height_threshold(self, value: float) -> None:
        if value < 0:
            raise GroundError(f"height_threshold must be non-negative, got {value}.")
        self._height_threshold = value
        self._cache.clear()

    def classify(
        self,
        cloud: PointCloud,
        **kwargs: Any,
    ) -> BoolArray1D:
        """
        Classify points as ground or non-ground.

        Parameters
        ----------
        cloud
            Input point cloud.
        **kwargs
            Additional arguments passed to the classifier.

        Returns
        -------
        BoolArray1D
            Boolean mask where True = ground.
        """
        classifier = self._get_classifier(**kwargs)
        return classifier.classify(cloud)

    def extract(
        self,
        cloud: PointCloud,
        **kwargs: Any,
    ) -> PointCloud:
        """Extract ground points."""
        extractor_class = self._EXTRACTORS.get(self._method)
        if extractor_class is not None:
            extractor = self._create_extractor(extractor_class, **kwargs)
            return extractor.extract(cloud)

        mask = self.classify(cloud, **kwargs)
        if not mask.any():
            raise GroundError(_NO_GROUND_POINTS_ERROR)
        return _build_ground_cloud_from_mask(cloud, mask)

    def _nearest_ground_elevation(
        self,
        cloud: PointCloud,
        mask: BoolArray1D,
    ) -> FloatArray1D:
        """Estimate elevation using the nearest classified ground point."""
        from topocore.processing.neighbors import NeighborhoodManager

        x, y, z = _extract_xyz(cloud)
        ground_indices = np.flatnonzero(mask)
        if ground_indices.size == 0:
            raise GroundError(_NO_GROUND_POINTS_ERROR)

        ground_points = np.column_stack((x[ground_indices], y[ground_indices], z[ground_indices]))
        manager = NeighborhoodManager.from_array(ground_points)
        elevations = np.empty(len(x), dtype=np.float64)

        for index in range(len(x)):
            indices, _ = manager.query_point(
                x[index],
                y[index],
                z[index],
                k=1,
            )
            elevations[index] = ground_points[indices[0], 2]

        return elevations

    def estimate_elevation(
        self,
        cloud: PointCloud,
        **kwargs: Any,
    ) -> FloatArray1D:
        """Estimate ground elevation for each point."""
        estimator_class = self._ELEVATION_ESTIMATORS.get(self._method)
        if estimator_class is not None:
            estimator = estimator_class(**self._get_params(**kwargs))
            return estimator.estimate(cloud)

        mask = self.classify(cloud, **kwargs)
        if not mask.any():
            raise GroundError(_NO_GROUND_POINTS_ERROR)
        return self._nearest_ground_elevation(cloud, mask)

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def _get_classifier(self, **kwargs: Any) -> GroundClassifier:
        classifier_class = self._SUPPORTED_METHODS[self._method]
        return classifier_class(**self._get_params(**kwargs))

    def _get_params(self, **kwargs: Any) -> dict[str, Any]:
        """Return constructor parameters for the selected implementation."""
        if self._method == "grid":
            parameters: dict[str, Any] = {
                "cell_size": self._cell_size,
                "height_threshold": self._height_threshold,
            }
        elif self._method == "adaptive_grid":
            parameters = {
                "base_cell_size": self._cell_size,
                "min_cell_size": self._min_cell_size,
                "max_cell_size": self._max_cell_size,
                "base_height_threshold": self._height_threshold,
                "slope_threshold": self._slope_threshold,
                "use_multiresolution": self._use_multiresolution,
            }
        elif self._method == "progressive_tin":
            parameters = {
                "cell_size": self._cell_size,
                "max_distance": self._max_distance,
                "max_angle": self._max_angle,
                "max_iterations": self._max_iterations,
            }
        elif self._method == "pmf":
            parameters = {
                "cell_size": self._cell_size,
                "initial_distance": self._pmf_initial_distance,
                "max_distance": self._pmf_max_distance,
                "slope": self._pmf_slope,
                "max_window_size": self._pmf_max_window_size,
                "window_base": self._pmf_window_base,
                "exponential": self._pmf_exponential,
                "max_grid_cells": self._pmf_max_grid_cells,
            }
        else:  # csf
            parameters = {
                "cloth_resolution": self._csf_cloth_resolution,
                "rigidness": self._csf_rigidness,
                "time_step": self._csf_time_step,
                "class_threshold": self._csf_class_threshold,
                "iterations": self._csf_iterations,
                "slope_smooth": self._csf_slope_smooth,
            }

        parameters.update(kwargs)
        return parameters

    def _create_extractor(
        self,
        extractor_class: type[GroundExtractor],
        **kwargs: Any,
    ) -> GroundExtractor:
        return extractor_class(**self._get_params(**kwargs))

    def __call__(
        self,
        cloud: PointCloud,
        **kwargs: Any,
    ) -> BoolArray1D:
        """Callable interface for classification."""
        return self.classify(cloud, **kwargs)


__all__ = ["GroundManager"]
