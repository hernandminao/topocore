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

from typing import Any, ClassVar, Protocol, TypeAlias

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
    select_at_indices,
    validate_viewpoint,
)
from .pca import PCANormalEstimator
from .weighted_pca import WeightedPCANormalEstimator

#: id(cloud) is computed fresh on every call (never stored on the
#: manager) -- see PR19 session notes: an earlier version of
#: topocore.processing.features.manager.FeatureManager stored
#: id(cloud) once, at construction, and silently returned stale
#: results for every other cloud. This module's cache was actually
#: never wired up at all before PR19 (a related but different bug:
#: harmless but wasteful, not silently wrong) -- being wired up now,
#: this key is designed from the start to avoid the features/manager.py
#: mistake.
#:
#: The viewpoint component is the actual (x, y, z) tuple, not merely
#: whether a viewpoint was given -- two DIFFERENT viewpoints must not
#: collide into the same cache entry. Likewise sigma (used only by
#: weighted_pca) is included explicitly; omitting it would let two
#: different sigma values silently share a cache entry, the same
#: category of bug as the viewpoint-identity mistake this replaces.
CacheKey: TypeAlias = tuple[
    int,
    int,
    str,
    int,
    bool,
    tuple[float, float, float] | None,
    float | None,
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
        "_cache",
        "_k",
        "_method",
        "_orient_upward",
        "_viewpoint",
    )

    _SUPPORTED_METHODS: ClassVar[
        dict[
            str,
            NormalEstimatorFactory,
        ]
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
        # PR21 remediation (NORMALS-MANAGER-001): __init__ previously
        # assigned viewpoint directly, bypassing the setter (and thus
        # any validation) entirely -- confirmed directly that
        # `NormalManager(viewpoint=(1, 2, 3))` was accepted silently
        # at construction, only to fail later with a confusing
        # AttributeError once actually used in _orient_normals().
        self._viewpoint: Vector3D | None = validate_viewpoint(viewpoint)

        self._cache: LRUCache[CacheKey, tuple[FloatArray2D, FloatArray1D]] = LRUCache(maxsize=cache_size)

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
        # PR21 remediation (NORMALS-MANAGER-001): previously did
        # `value.shape != (3,)` directly, assuming `value` already
        # had a `.shape` attribute -- a plain tuple/list would raise
        # a confusing AttributeError here instead of NormalError,
        # the same root gap already found and fixed for
        # PCANormalEstimator/WeightedPCANormalEstimator's own
        # constructors (PCA-VIEWPOINT-001). Reuses the same shared
        # validate_viewpoint() helper for a consistent contract.
        self._viewpoint = validate_viewpoint(value)
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
        normals, _ = self._estimate_both_cached(
            cloud,
            manager,
            **kwargs,
        )

        return normals

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
        normals, _ = self._estimate_both_cached(
            cloud,
            manager,
            **kwargs,
        )

        return select_at_indices(normals, indices)

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
        estimator, _ = self._get_estimator(**kwargs)

        if isinstance(
            estimator,
            NormalAndCurvatureEstimator,
        ):
            _, curvature = self._estimate_both_cached(
                cloud,
                manager,
                **kwargs,
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
        return self._estimate_both_cached(
            cloud,
            manager,
            **kwargs,
        )

    def clear_cache(
        self,
    ) -> None:
        """Clear cache."""
        self._cache.clear()

    def _estimate_both_cached(
        self,
        cloud: PointCloud,
        manager: NeighborhoodManager | None,
        **kwargs: Any,
    ) -> tuple[FloatArray2D, FloatArray1D]:
        """
        Shared, cached (normals, curvature) computation used by
        ``estimate()``/``estimate_at()``/``estimate_curvature()``/
        ``estimate_both()`` alike, so calling more than one of them
        with the same cloud and parameters computes the underlying
        PCA only once.

        Note: if ``manager`` (an explicitly-provided
        ``NeighborhoodManager``) is given, it is NOT itself part of
        the cache key -- only ``id(cloud)``, ``cloud.version``, and
        the resolved estimation parameters are. Passing a different,
        differently configured ``NeighborhoodManager`` for the same
        cloud, version, and parameters is an advanced, rare override
        this cache does not specially detect; ``clear_cache()`` is
        available if that matters for a specific workflow.
        """
        estimator, params = self._get_estimator(**kwargs)

        cache_key = self._cache_key(
            id(cloud),
            cloud.version,
            self._method,
            params,
        )

        cached = self._cache.get(cache_key)

        if cached is not None:
            return cached

        if isinstance(
            estimator,
            NormalAndCurvatureEstimator,
        ):
            result = estimator.estimate_both(
                cloud,
                manager=manager,
            )
        else:
            normals = estimator.estimate(
                cloud,
                manager=manager,
            )

            pca = PCANormalEstimator(
                k=params["k"],
                orient_upward=params["orient_upward"],
                viewpoint=params["viewpoint"],
            )

            _, curvature = pca.estimate_both(
                cloud,
                manager=manager,
            )

            result = (normals, curvature)

        self._cache.set(
            cache_key,
            result,
        )

        return result

    def _get_estimator(
        self,
        **kwargs: Any,
    ) -> tuple[NormalEstimator, dict[str, Any]]:
        """
        Create the current estimator, along with the resolved
        parameters used to build it (needed by ``_cache_key()`` --
        the actual per-call values, after ``kwargs`` overrides, not
        just the manager's stored defaults).
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

        return factory(**params), params

    def _cache_key(
        self,
        cloud_id: int,
        cloud_version: int,
        method: str,
        params: dict[str, Any],
    ) -> CacheKey:
        """
        Generate cache key from the actual resolved parameters.

        PR21.3.1: includes ``cloud_version`` alongside ``cloud_id``
        (formerly the sole key component). Found and fixed as a
        real, demonstrated bug: ``id(cloud)`` alone cannot detect a
        ``PointCloud`` mutated in place (via ``add_chunk()``/
        ``remove_chunk()``/``clear()``) between two calls on the
        same ``NormalManager`` instance, since Python object
        identity never changes when an object is mutated rather than
        replaced -- confirmed directly: mutating a flat-plane cloud
        into a steeply-tilted one via ``add_chunk()`` and re-calling
        ``estimate()`` on the SAME manager returned the stale,
        pre-mutation normal instead of recomputing. ``PointCloud.
        version`` (a counter incremented by exactly those three
        mutating methods) closes this gap. See that property's own
        docstring for its one documented limitation (it cannot detect
        a caller reaching into a ``Chunk``'s array and mutating it
        in place directly, bypassing ``PointCloud``'s own methods
        entirely).
        """
        viewpoint = params["viewpoint"]
        viewpoint_key = (
            (float(viewpoint[0]), float(viewpoint[1]), float(viewpoint[2])) if viewpoint is not None else None
        )

        sigma = params.get("sigma")

        return (
            cloud_id,
            cloud_version,
            method,
            int(params["k"]),
            bool(params["orient_upward"]),
            viewpoint_key,
            float(sigma) if sigma is not None else None,
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
