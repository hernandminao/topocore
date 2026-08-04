"""
topocore.processing.ground.pmf
================================

Progressive Morphological Filter (PMF) ground classification.

The implementation uses a compact raster whose memory usage depends on the
covered terrain extent rather than on the number of input points. PointCloud
chunks are scanned incrementally, so X/Y/Z arrays for the complete cloud are
never materialized at the same time.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import override

import numpy as np
from scipy import ndimage

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import GroundError
from topocore.processing.types import BoolArray1D

from .base import GroundClassifier, GroundExtractor


_EMPTY_CLOUD_ERROR = "Cannot classify an empty point cloud."
_NO_GROUND_POINTS_ERROR = "No ground points found. Try adjusting PMF parameters."


@dataclass(frozen=True, slots=True)
class _GridLayout:
    min_x: float
    min_y: float
    rows: int
    columns: int
    point_count: int

    @property
    def cell_count(self) -> int:
        return self.rows * self.columns


def _validate_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise GroundError(f"{name} must be positive and finite, got {value}.")


def _validate_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise GroundError(f"{name} must be non-negative and finite, got {value}.")


def _scan_grid_layout(
    cloud: PointCloud,
    cell_size: float,
) -> _GridLayout:
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    point_count = 0

    for chunk in cloud:
        x = np.asarray(chunk[PointAttribute.X])
        y = np.asarray(chunk[PointAttribute.Y])
        z = np.asarray(chunk[PointAttribute.Z])

        if x.size != y.size or x.size != z.size:
            raise GroundError("Point cloud coordinate attributes must have equal lengths.")
        if x.size == 0:
            continue
        if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
            raise GroundError("Point cloud coordinates must contain only finite values.")

        min_x = min(min_x, float(np.min(x)))
        min_y = min(min_y, float(np.min(y)))
        max_x = max(max_x, float(np.max(x)))
        max_y = max(max_y, float(np.max(y)))
        point_count += int(x.size)

    if point_count == 0:
        raise GroundError(_EMPTY_CLOUD_ERROR)

    columns = int(np.floor((max_x - min_x) / cell_size)) + 1
    rows = int(np.floor((max_y - min_y) / cell_size)) + 1

    return _GridLayout(
        min_x=min_x,
        min_y=min_y,
        rows=rows,
        columns=columns,
        point_count=point_count,
    )


def _cell_indices(
    x: np.ndarray,
    y: np.ndarray,
    layout: _GridLayout,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    columns = np.floor((x - layout.min_x) / cell_size).astype(np.intp)
    rows = np.floor((y - layout.min_y) / cell_size).astype(np.intp)

    # Guard against a last-cell overflow caused by floating-point rounding.
    np.clip(columns, 0, layout.columns - 1, out=columns)
    np.clip(rows, 0, layout.rows - 1, out=rows)
    return rows, columns


def _build_minimum_surface(
    cloud: PointCloud,
    layout: _GridLayout,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    surface = np.full(
        (layout.rows, layout.columns),
        np.inf,
        dtype=np.float64,
    )
    flat_surface = surface.ravel()

    for chunk in cloud:
        x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
        y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
        z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)
        if x.size == 0:
            continue

        rows, columns = _cell_indices(x, y, layout, cell_size)
        flat_indices = rows * layout.columns + columns
        np.minimum.at(flat_surface, flat_indices, z)

    occupied = np.isfinite(surface)
    if not occupied.any():
        raise GroundError(_EMPTY_CLOUD_ERROR)

    if occupied.all():
        return surface, occupied

    # Fill empty cells once with their nearest occupied-cell value. This keeps
    # the morphology stable without retaining point-sized intermediate arrays.
    nearest = ndimage.distance_transform_edt(
        ~occupied,
        return_distances=False,
        return_indices=True,
    )
    filled = surface[tuple(nearest)]
    return np.asarray(filled, dtype=np.float64), occupied


def _window_sizes(
    max_window_size: int,
    window_base: float,
    exponential: bool,
) -> tuple[int, ...]:
    maximum = max_window_size if max_window_size % 2 == 1 else max_window_size - 1
    if maximum < 3:
        return (3,)

    if not exponential:
        return tuple(range(3, maximum + 1, 2))

    windows: list[int] = []
    exponent = 0
    while True:
        radius = max(1, int(round(window_base**exponent)))
        size = min(maximum, 2 * radius + 1)
        if not windows or size > windows[-1]:
            windows.append(size)
        if size >= maximum:
            break
        exponent += 1

    return tuple(windows)


def _progressive_filter(
    surface: np.ndarray,
    occupied: np.ndarray,
    windows: tuple[int, ...],
    cell_size: float,
    initial_distance: float,
    max_distance: float,
    slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    terrain = surface.copy()
    ground_cells = occupied.copy()
    previous_radius = 0

    for window_size in windows:
        radius = window_size // 2
        opened = ndimage.grey_opening(
            terrain,
            size=(window_size, window_size),
            mode="nearest",
        )
        threshold = min(
            max_distance,
            initial_distance + slope * cell_size * (radius - previous_radius),
        )

        ground_cells &= (terrain - opened) <= threshold
        np.copyto(terrain, opened, where=~ground_cells)
        previous_radius = radius

    return terrain, ground_cells


def _build_ground_cloud_from_mask_streaming(
    cloud: PointCloud,
    mask: BoolArray1D,
) -> PointCloud:
    """Build the result chunk by chunk without flattening every attribute."""
    from topocore.pointcloud.chunk import Chunk

    result = PointCloud()
    attributes = list(cloud.attributes)
    offset = 0

    for source in cloud:
        chunk_size = len(source[PointAttribute.X])
        chunk_mask = mask[offset : offset + chunk_size]
        offset += chunk_size
        selected_count = int(np.count_nonzero(chunk_mask))
        if selected_count == 0:
            continue

        target = Chunk(size=selected_count, attributes=attributes)
        for attribute in attributes:
            target[attribute][:] = source[attribute][chunk_mask]
        result.add_chunk(target)

    if offset != mask.size:
        raise GroundError(f"Ground mask length {mask.size} does not match point count {offset}.")

    result.update_bounds()
    return result


class PMFGroundClassifier(GroundClassifier):
    """
    Progressive Morphological Filter ground classifier.

    Parameters
    ----------
    cell_size
        Raster cell size used to compute minimum elevations.
    initial_distance
        Initial elevation threshold above the opened terrain surface.
    max_distance
        Maximum elevation threshold used by progressive iterations and final
        point classification.
    slope
        Threshold growth per horizontal unit.
    max_window_size
        Maximum odd morphological window size in cells.
    window_base
        Exponential growth base for window radii.
    exponential
        Use exponentially growing windows when True, otherwise use all odd
        window sizes from 3 through max_window_size.
    max_grid_cells
        Safety limit for the compact terrain raster. The limit is independent
        of point count and prevents accidental allocations caused by an
        extremely large or sparse coordinate extent.
    """

    __slots__ = (
        "_cell_size",
        "_initial_distance",
        "_max_distance",
        "_slope",
        "_max_window_size",
        "_window_base",
        "_exponential",
        "_max_grid_cells",
    )

    def __init__(
        self,
        cell_size: float = 1.0,
        initial_distance: float = 0.15,
        max_distance: float = 2.5,
        slope: float = 1.0,
        max_window_size: int = 33,
        window_base: float = 2.0,
        exponential: bool = True,
        max_grid_cells: int = 8_000_000,
    ) -> None:
        _validate_positive("cell_size", cell_size)
        _validate_non_negative("initial_distance", initial_distance)
        _validate_positive("max_distance", max_distance)
        _validate_non_negative("slope", slope)
        _validate_positive("window_base", window_base)

        if initial_distance > max_distance:
            raise GroundError(f"initial_distance must be <= max_distance, got {initial_distance} > {max_distance}.")
        if (
            isinstance(max_window_size, bool)
            or not isinstance(max_window_size, (int, np.integer))
            or max_window_size < 3
        ):
            raise GroundError(f"max_window_size must be an integer >= 3, got {max_window_size}.")
        if not isinstance(exponential, bool):
            raise GroundError(f"exponential must be a bool, got {type(exponential).__name__}.")
        if window_base <= 1.0 and exponential:
            raise GroundError(f"window_base must be > 1 when exponential=True, got {window_base}.")
        if isinstance(max_grid_cells, bool) or not isinstance(max_grid_cells, (int, np.integer)) or max_grid_cells < 1:
            raise GroundError(f"max_grid_cells must be an integer >= 1, got {max_grid_cells}.")

        self._cell_size = float(cell_size)
        self._initial_distance = float(initial_distance)
        self._max_distance = float(max_distance)
        self._slope = float(slope)
        self._max_window_size = int(max_window_size)
        self._window_base = float(window_base)
        self._exponential = bool(exponential)
        self._max_grid_cells = int(max_grid_cells)

    @override
    def classify(self, cloud: PointCloud) -> BoolArray1D:
        """Classify points as ground without concatenating cloud attributes."""
        layout = _scan_grid_layout(cloud, self._cell_size)
        if layout.cell_count > self._max_grid_cells:
            minimum_cell_size = sqrt(
                (layout.rows * self._cell_size) * (layout.columns * self._cell_size) / self._max_grid_cells
            )
            suggested = max(self._cell_size, minimum_cell_size)
            raise GroundError(
                "PMF compact grid would contain "
                f"{layout.cell_count} cells, exceeding max_grid_cells="
                f"{self._max_grid_cells}. Increase cell_size to approximately "
                f"{suggested:.6g} or raise max_grid_cells."
            )

        surface, occupied = _build_minimum_surface(
            cloud,
            layout,
            self._cell_size,
        )
        terrain, ground_cells = _progressive_filter(
            surface,
            occupied,
            _window_sizes(
                self._max_window_size,
                self._window_base,
                self._exponential,
            ),
            self._cell_size,
            self._initial_distance,
            self._max_distance,
            self._slope,
        )

        mask = np.zeros(layout.point_count, dtype=np.bool_)
        offset = 0
        for chunk in cloud:
            x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
            y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
            z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)
            if x.size == 0:
                continue

            rows, columns = _cell_indices(x, y, layout, self._cell_size)
            count = int(x.size)
            residual = z - terrain[rows, columns]
            mask[offset : offset + count] = ground_cells[rows, columns] & (residual <= self._max_distance)
            offset += count

        return mask

    @override
    def name(self) -> str:
        return "pmf"


class PMFGroundExtractor(GroundExtractor):
    """Extract ground points using :class:`PMFGroundClassifier`."""

    __slots__ = ("_classifier",)

    def __init__(
        self,
        cell_size: float = 1.0,
        initial_distance: float = 0.15,
        max_distance: float = 2.5,
        slope: float = 1.0,
        max_window_size: int = 33,
        window_base: float = 2.0,
        exponential: bool = True,
        max_grid_cells: int = 8_000_000,
    ) -> None:
        self._classifier = PMFGroundClassifier(
            cell_size=cell_size,
            initial_distance=initial_distance,
            max_distance=max_distance,
            slope=slope,
            max_window_size=max_window_size,
            window_base=window_base,
            exponential=exponential,
            max_grid_cells=max_grid_cells,
        )

    @override
    def extract(self, cloud: PointCloud) -> PointCloud:
        mask = self._classifier.classify(cloud)
        if not mask.any():
            raise GroundError(_NO_GROUND_POINTS_ERROR)
        return _build_ground_cloud_from_mask_streaming(cloud, mask)

    @override
    def name(self) -> str:
        return "pmf"


__all__ = [
    "PMFGroundClassifier",
    "PMFGroundExtractor",
]
