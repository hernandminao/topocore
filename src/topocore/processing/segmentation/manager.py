from __future__ import annotations

from typing import Any

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError

from .base import SegmentationResult, Segmenter
from .connected_components import ConnectedComponentsSegmenter
from .dbscan import DBSCANSegmenter
from .region_growing import RegionGrowingSegmenter
from .specific import BuildingSegmenter, TreeSegmenter


class SegmentationManager:
    """
    High-level manager for point cloud segmentation.

    This class provides a unified interface for segmentation with
    automatic method selection.

    Supported methods:
    - "dbscan": DBSCAN clustering
    - "region_growing": Region growing
    - "connected_components": Connected components
    - "trees": Specialized tree segmentation
    - "buildings": Specialized building segmentation

    Examples
    --------
    >>> manager = SegmentationManager(method="region_growing")
    >>> result = manager.segment(cloud)
    >>> segments = result.get_segments()

    >>> manager = SegmentationManager(method="trees", min_height=0.5)
    >>> result = manager.segment(forest_cloud)
    >>> trees = result.get_segments()
    """

    __slots__ = (
        "_method",
        "_params",
    )

    _SUPPORTED_METHODS = frozenset(
        {
            "dbscan",
            "region_growing",
            "connected_components",
            "trees",
            "buildings",
        }
    )

    def __init__(
        self,
        method: str = "region_growing",
        **kwargs: Any,
    ) -> None:
        if method not in self._SUPPORTED_METHODS:
            self._validate_method(method)

        self._method = method
        self._params = kwargs

    @property
    def method(self) -> str:
        """Get the current method."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """Set the method."""
        if value not in self._SUPPORTED_METHODS:
            self._validate_method(value)
        self._method = value

    def set_params(
        self,
        **kwargs: Any,
    ) -> None:
        """Set parameters for the current method."""
        self._params.update(kwargs)

    def segment(
        self,
        cloud: PointCloud,
        **kwargs: Any,
    ) -> SegmentationResult:
        """
        Segment the point cloud.

        Parameters
        ----------
        cloud
            Input point cloud.
        **kwargs
            Additional arguments passed to the segmenter.

        Returns
        -------
        SegmentationResult
            Result containing labels and segment information.

        Raises
        ------
        SegmentationError
            If segmentation fails.
        """
        if cloud.is_empty:
            raise SegmentationError("Cannot segment an empty point cloud.")

        segmenter = self._create_segmenter(**kwargs)
        return segmenter.segment(cloud)

    def _create_segmenter(
        self,
        **kwargs: Any,
    ) -> Segmenter:
        """Create the appropriate segmenter instance."""
        params = self._params.copy()
        params.update(kwargs)
        p = params

        if self._method == "dbscan":
            return DBSCANSegmenter(
                eps=p.get("eps", 0.5),
                min_samples=p.get("min_samples", 5),
                use_adaptive_eps=p.get("use_adaptive_eps", False),
                include_noise=p.get("include_noise", True),
            )

        if self._method == "region_growing":
            return RegionGrowingSegmenter(
                k=p.get("k", 10),
                curvature_threshold=p.get("curvature_threshold", 0.05),
                normal_angle_threshold=p.get("normal_angle_threshold", 15.0),
                min_region_size=p.get("min_region_size", 10),
                max_region_size=p.get("max_region_size", 1_000_000),
                use_adaptive_k=p.get("use_adaptive_k", False),
            )

        if self._method == "connected_components":
            return ConnectedComponentsSegmenter(
                distance_threshold=p.get("distance_threshold", 0.5),
                min_points=p.get("min_points", 10),
                use_adaptive_threshold=p.get("use_adaptive_threshold", False),
            )

        if self._method == "trees":
            return TreeSegmenter(
                min_height=p.get("min_height", 0.5),
                max_height=p.get("max_height", 50.0),
                eps=p.get("eps", 0.5),
                min_samples=p.get("min_samples", 5),
                min_points_per_tree=p.get("min_points_per_tree", 10),
                ground_method=p.get("ground_method", "grid"),
            )

        if self._method == "buildings":
            return BuildingSegmenter(
                min_height=p.get("min_height", 1.0),
                max_height=p.get("max_height", 100.0),
                k=p.get("k", 10),
                curvature_threshold=p.get("curvature_threshold", 0.02),
                normal_angle_threshold=p.get("normal_angle_threshold", 10.0),
                min_points_per_building=p.get("min_points_per_building", 100),
                ground_method=p.get("ground_method", "grid"),
            )

        raise RuntimeError(f"Unexpected segmentation method: {self._method}")

    @classmethod
    def _validate_method(cls, method: str) -> None:
        if method not in cls._SUPPORTED_METHODS:
            supported = ", ".join(sorted(cls._SUPPORTED_METHODS))
            raise SegmentationError(f"Unsupported method '{method}'. Supported methods: {supported}.")

    def __call__(
        self,
        cloud: PointCloud,
        **kwargs: Any,
    ) -> SegmentationResult:
        """Callable interface."""
        return self.segment(cloud, **kwargs)


__all__ = [
    "SegmentationManager",
]
