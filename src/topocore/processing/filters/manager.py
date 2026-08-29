"""
topocore.processing.filters.manager
===================================

Filter manager for chaining and applying filters.

This module provides a high-level manager that orchestrates the
application of multiple filters in sequence. The manager handles:
- Filter registration
- Filter chaining (pipeline execution)
- Caching of intermediate results
- Statistics tracking

The manager follows the Chain of Responsibility pattern: each filter
is applied in sequence, and the output of one filter becomes the
input to the next.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.bbox3d import BBox3D
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import (
    build_cloud_from_mask,
    flatten_attributes,
)
from topocore.processing.cache import LRUCache
from topocore.processing.exceptions import FilterError
from topocore.processing.types import BoolArray1D

from .base import Filter
from .clip_polygon import ClipPolygonFilter
from .crop_box import CropBoxFilter
from .pass_through import Axis, PassThroughFilter
from .radius import RadiusOutlierFilter
from .statistical import StatisticalOutlierFilter

type FilterStats = dict[str, Any]
type CacheKey = tuple[int, int, str]


def _cache_key_belongs_to(cloud_id: int, key: CacheKey) -> bool:
    """Predicate for LRUCache.remove_where(): matches every cache entry whose stage-input cloud id is `cloud_id`."""
    return key[0] == cloud_id


type PolygonArray = NDArray[np.float64]


class FilterManager:
    """
    High-level manager for filter chaining and application.

    This class provides a unified interface for applying multiple
    filters in sequence. Filters are executed in the order they
    are added.

    Examples
    --------
    >>> manager = FilterManager()
    >>> manager.add(StatisticalOutlierFilter(k=8, std_ratio=1.0))
    >>> manager.add(PassThroughFilter(Axis.Z, min_value=0.0, max_value=100.0))
    >>> filtered_cloud = manager.apply(cloud)

    >>> # Or use method chaining
    >>> filtered_cloud = (FilterManager()
    ...     .add(StatisticalOutlierFilter(k=8, std_ratio=1.0))
    ...     .add(PassThroughFilter(Axis.Z, min_value=0.0, max_value=100.0))
    ...     .apply(cloud))
    """

    __slots__ = (
        "_filters",
        "_cache",
        "_track_stats",
        "_last_stats",
    )

    def __init__(
        self,
        track_stats: bool = True,
        cache_size: int = 8,
    ) -> None:

        self._filters: list[Filter] = []

        self._cache: LRUCache[
            CacheKey,
            PointCloud,
        ] = LRUCache(
            maxsize=cache_size,
        )

        self._track_stats = track_stats

        self._last_stats: list[FilterStats] | None = None

    def add(
        self,
        filter: Filter,
    ) -> FilterManager:

        self._filters.append(filter)

        return self

    def add_statistical_outlier(
        self,
        k: int = 8,
        std_ratio: float = 1.0,
        min_points: int = 3,
        strict: bool = False,
    ) -> FilterManager:

        return self.add(
            StatisticalOutlierFilter(
                k,
                std_ratio,
                min_points,
                strict,
            )
        )

    def add_radius_outlier(
        self,
        radius: float = 1.0,
        min_neighbors: int = 4,
        include_self: bool = True,
        min_points: int = 3,
        strict: bool = False,
    ) -> FilterManager:

        return self.add(
            RadiusOutlierFilter(
                radius,
                min_neighbors,
                include_self,
                min_points,
                strict,
            )
        )

    def add_pass_through(
        self,
        axis: Axis,
        min_value: float,
        max_value: float,
    ) -> FilterManager:

        return self.add(
            PassThroughFilter(
                axis,
                min_value,
                max_value,
            )
        )

    def add_crop_box(
        self,
        box: "BBox3D",
    ) -> FilterManager:

        return self.add(CropBoxFilter(box))

    def add_clip_polygon(
        self,
        polygon: PolygonArray,
    ) -> FilterManager:

        return self.add(ClipPolygonFilter(polygon))

    def apply(
        self,
        cloud: PointCloud,
    ) -> PointCloud:

        if not self._filters:
            return cloud

        current = cloud

        stats: list[FilterStats] = []

        for index, filter in enumerate(self._filters):
            # PR21 remediation (FILTER-MANAGER-001): `cache_key`'s
            # first element identifies the INPUT cloud to this stage
            # (`current`, before the reassignment below) -- captured
            # here in `stage_input` so the finalizer registered after
            # a cache miss attaches to the correct object, not to
            # `current`'s later reassignment to this stage's output.
            stage_input = current
            cache_key: CacheKey = (
                id(stage_input),
                index,
                filter.name(),
            )

            cached = self._cache.get(cache_key)

            if cached is not None:
                current = cached

                if self._track_stats:
                    stats.append(
                        {
                            "filter": filter.name(),
                            "cached": True,
                        }
                    )

                continue

            try:
                before = current.point_count

                current = filter.apply(current)

                after = current.point_count

                if self._track_stats:
                    removed = before - after

                    stats.append(
                        {
                            "filter": filter.name(),
                            "points_before": before,
                            "points_after": after,
                            "removed": removed,
                            "removed_percent": (100.0 * removed / before if before > 0 else 0.0),
                        }
                    )

                self._cache.set(
                    cache_key,
                    current,
                )

                # PR21 remediation (FILTER-MANAGER-001): apply()'s
                # own cache persists across separate apply() calls
                # (see class docstring's own reuse pattern), and
                # previously keyed cache entries solely by
                # id(stage_input) -- a value CPython is well known to
                # reuse for a LATER, unrelated object once
                # `stage_input` is garbage collected. Confirmed
                # deterministically (bypassing the non-deterministic
                # nature of actual allocator reuse) by directly
                # inserting a cache entry under the exact id()-based
                # key a colliding object would produce, and
                # confirming apply() blindly returned it -- a stale,
                # wrong result from a completely different, already
                # -freed PointCloud.
                #
                # Fix: a weakref finalizer is registered on
                # `stage_input` itself (not merely its id) that
                # proactively purges every cache entry tied to this
                # exact id() the MOMENT this object is garbage
                # collected -- synchronously, not lazily. This means
                # a later object reusing the same freed address can
                # never find a leftover stale entry: it was already
                # removed before the reuse could ever happen.
                #
                # This preserves the cache's existing global LRUCache
                # eviction bound (cache_size keeps meaning "N entries
                # total", not "N entries per cloud") -- only cleanup
                # timing changed, not the eviction policy, the key
                # structure, or any public API.
                weakref.finalize(
                    stage_input,
                    self._cache.remove_where,
                    functools.partial(_cache_key_belongs_to, id(stage_input)),
                )

            except Exception as exc:
                raise FilterError(f"Filter '{filter.name()}' failed: {exc}") from exc

        if self._track_stats:
            self._last_stats = stats

        return current

    def apply_masks(
        self,
        cloud: PointCloud,
    ) -> list[BoolArray1D]:

        masks: list[BoolArray1D] = []

        current = cloud

        for filter in self._filters:
            mask = filter.mask(current)

            masks.append(mask)

            flattened = flatten_attributes(current)

            current = build_cloud_from_mask(
                flattened,
                mask,
            )

        return masks

    def clear(
        self,
    ) -> None:

        self._filters.clear()
        self._cache.clear()
        self._last_stats = None

    def statistics(
        self,
    ) -> list[FilterStats] | None:

        return self._last_stats

    @property
    def filter_count(
        self,
    ) -> int:

        return len(self._filters)

    @property
    def filters(
        self,
    ) -> list[Filter]:

        return self._filters.copy()

    def __len__(
        self,
    ) -> int:

        return len(self._filters)

    def __iter__(
        self,
    ) -> Iterator[Filter]:

        return iter(self._filters)

    def __getitem__(
        self,
        index: int,
    ) -> Filter:

        return self._filters[index]


__all__ = [
    "FilterManager",
]
