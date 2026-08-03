"""
topocore.processing.filters
===========================

Point cloud filtering.

This package provides various filters for point cloud processing:

- Statistical Outlier Removal (SOR): removes statistically anomalous points
- Radius Outlier Removal (ROR): removes points with low local density
- Pass-Through: crops points by coordinate range on a single axis
- Crop Box: crops points inside a 3D bounding box
- Clip Polygon: crops points inside a 2D polygon (planimetric)

Filters can be chained together using the FilterManager.

Public API
----------
- Filter: abstract base class for filters
- BinaryFilter: base class for filters that produce a binary mask
- StatisticalOutlierFilter: SOR filter
- RadiusOutlierFilter: ROR filter
- PassThroughFilter: pass-through filter
- CropBoxFilter: crop box filter
- ClipPolygonFilter: clip polygon filter
- FilterManager: high-level manager for filter chaining
- Axis: coordinate axis enum (X, Y, Z)

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import BinaryFilter, Filter
from .clip_polygon import ClipPolygonFilter, point_in_polygon
from .crop_box import CropBoxFilter
from .manager import FilterManager
from .pass_through import Axis, PassThroughFilter
from .radius import RadiusOutlierFilter
from .statistical import StatisticalOutlierFilter

__all__ = [
    "Filter",
    "BinaryFilter",
    "StatisticalOutlierFilter",
    "RadiusOutlierFilter",
    "PassThroughFilter",
    "Axis",
    "CropBoxFilter",
    "ClipPolygonFilter",
    "point_in_polygon",
    "FilterManager",
]
