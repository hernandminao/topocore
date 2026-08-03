"""
topocore.processing.normals.manager
===================================

Normal estimation manager with automatic method selection.

This module provides a high-level manager that selects the appropriate
normal estimation method based on the point cloud characteristics
and user preferences.

The manager supports:
- Automatic method selection
- Method switching
- Caching of computed normals

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.cache import LRUCache
from topocore.processing.exceptions import NormalError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import (
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
    Vector3D,
)

from .base import (
    CurvatureEstimator,
    NormalAndCurvatureEstimator,
    NormalEstimator,
)
from .pca import PCANormalEstimator
from .weighted_pca import WeightedPCANormalEstimator

CacheKey: TypeAlias = tuple[
    int,
    str,
    int,
    bool,
    bool,
]


class NormalEstimatorFactory(Protocol):
    """
    Factory interface for normal estimator creation.
    """

    def __call__(
        self,
        **kwargs: Any,
    ) -> NormalEstimator: ...


def _create_pca(
    **kwargs: Any,
) -> NormalEstimator:
    """
    Create PCA normal estimator.
    """
    return PCANormalEstimator(
        k=kwargs["k"],
        orient_upward=kwargs["orient_upward"],
        viewpoint=kwargs["viewpoint"],
    )


def _create_weighted_pca(
    **kwargs: Any,
) -> NormalEstimator:
    """
    Create weighted PCA normal estimator.
    """
    return WeightedPCANormalEstimator(
        k=kwargs["k"],
        sigma=kwargs.get("sigma"),
        orient_upward=kwargs["orient_upward"],
        viewpoint=kwargs["viewpoint"],
    )


class NormalManager:
    """
    High-level manager for normal and curvature estimation.

    This class provides a unified interface for normal estimation
    with automatic method selection and caching.
    """

    __slots__ = (
        "_method",
        "_k",
        "_orient_upward",
        "_viewpoint",
        "_cache",
        "_cloud_id",
    )

    _SUPPORTED_METHODS: dict[
        str,
        NormalEstimatorFactory,
    ] = {
        "pca": _create_pca,
        "weighted_pca": _create_weighted_pca,
    }

    def __init__(
        self,
        method: str = "pca",
        k: int = 10,
        orient_upward: bool = True,
        viewpoint: Vector3D | None = None,
        cache_size: int = 16,
    ) -> None:

        if method not in self._SUPPORTED_METHODS:
            raise NormalError(f"Unsupported method: {method}. Supported: {list(self._SUPPORTED_METHODS)}")

        if k < 3:
            raise NormalError(f"k must be at least 3, got {k}.")

        self._method: str = method
        self._k: int = k
        self._orient_upward: bool = orient_upward
        self._viewpoint: Vector3D | None = viewpoint

        self._cache: LRUCache[CacheKey, Any] = LRUCache(maxsize=cache_size)

        self._cloud_id: int = 0

    @property
    def method(
        self,
    ) -> str:
        """Get current method."""
        return self._method

    @method.setter
    def method(
        self,
        value: str,
    ) -> None:
        """Set current method."""
        if value not in self._SUPPORTED_METHODS:
            raise NormalError(f"Unsupported method: {value}. Supported: {list(self._SUPPORTED_METHODS)}")

        self._method = value
        self._cache.clear()

    @property
    def k(
        self,
    ) -> int:
        """Get neighborhood size."""
        return self._k

    @k.setter
    def k(
        self,
        value: int,
    ) -> None:

        if value < 3:
            raise NormalError(f"k must be at least 3, got {value}.")

        self._k = value
        self._cache.clear()

    @property
    def orient_upward(
        self,
    ) -> bool:
        """Get orientation setting."""
        return self._orient_upward

    @orient_upward.setter
    def orient_upward(
        self,
        value: bool,
    ) -> None:

        self._orient_upward = value
        self._cache.clear()

    @property
    def viewpoint(
        self,
    ) -> Vector3D | None:
        """Get viewpoint."""
        return self._viewpoint

    @viewpoint.setter
    def viewpoint(
        self,
        value: Vector3D | None,
    ) -> None:

        if value is not None and value.shape != (3,):
            raise NormalError(f"viewpoint must have shape (3,), got {value.shape}.")

        self._viewpoint = value
        self._cache.clear()

    def estimate(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
        **kwargs: Any,
    ) -> FloatArray2D:
        """
        Estimate normals.
        """
        estimator = self._get_estimator(**kwargs)

        return estimator.estimate(
            cloud,
            manager=manager,
        )

    def estimate_at(
        self,
        cloud: PointCloud,
        indices: IntArray1D | None = None,
        *,
        manager: NeighborhoodManager | None = None,
        **kwargs: Any,
    ) -> FloatArray2D:
        """
        Estimate normals for selected points.
        """
        estimator = self._get_estimator(**kwargs)

        return estimator.estimate_at(
            cloud,
            indices,
            manager=manager,
        )

    def estimate_curvature(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
        **kwargs: Any,
    ) -> FloatArray1D:
        """
        Estimate curvature.
        """
        estimator = self._get_estimator(**kwargs)

        if isinstance(
            estimator,
            NormalAndCurvatureEstimator,
        ):
            _, curvature = estimator.estimate_both(
                cloud,
                manager=manager,
            )
            return curvature

        if isinstance(
            estimator,
            CurvatureEstimator,
        ):
            return estimator.estimate(
                cloud,
                manager=manager,
            )

        raise NormalError(f"{type(estimator).__name__} does not support curvature.")

    def estimate_both(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
        **kwargs: Any,
    ) -> tuple[FloatArray2D, FloatArray1D]:
        """
        Estimate normals and curvature.
        """
        estimator = self._get_estimator(**kwargs)

        if isinstance(
            estimator,
            NormalAndCurvatureEstimator,
        ):
            return estimator.estimate_both(
                cloud,
                manager=manager,
            )

        normals = estimator.estimate(
            cloud,
            manager=manager,
        )

        pca = PCANormalEstimator(
            k=self._k,
            orient_upward=self._orient_upward,
            viewpoint=self._viewpoint,
        )

        _, curvature = pca.estimate_both(
            cloud,
            manager=manager,
        )

        return normals, curvature

    def clear_cache(
        self,
    ) -> None:
        """Clear cache."""
        self._cache.clear()

    def _get_estimator(
        self,
        **kwargs: Any,
    ) -> NormalEstimator:
        """
        Create current estimator.
        """

        params: dict[str, Any] = {
            "k": kwargs.get(
                "k",
                self._k,
            ),
            "orient_upward": kwargs.get(
                "orient_upward",
                self._orient_upward,
            ),
            "viewpoint": kwargs.get(
                "viewpoint",
                self._viewpoint,
            ),
        }

        if "sigma" in kwargs:
            params["sigma"] = kwargs["sigma"]

        factory = self._SUPPORTED_METHODS[self._method]

        return factory(**params)

    def _cache_key(
        self,
        cloud_id: int,
        method: str,
        k: int,
        orient_upward: bool,
        viewpoint: Vector3D | None,
    ) -> CacheKey:
        """
        Generate cache key.
        """
        return (
            cloud_id,
            method,
            k,
            orient_upward,
            viewpoint is not None,
        )

    def __call__(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
        **kwargs: Any,
    ) -> FloatArray2D:
        """
        Callable interface.
        """
        return self.estimate(
            cloud,
            manager=manager,
            **kwargs,
        )


__all__ = [
    "NormalManager",
]
