"""
topocore.processing.segmentation
================================

Point cloud segmentation.

This package provides various methods for segmenting point clouds:

- DBSCAN: Density-based clustering
- Region Growing: Surface-based region growing
- Connected Components: Simple spatial clustering
- Trees: Specialized tree segmentation
- Buildings: Specialized building segmentation

Segmentation is essential for:
- Object extraction (trees, buildings, poles)
- Feature recognition
- Scene understanding
- 3D modeling

Public API
----------
- SegmentationResult: Result of a segmentation operation
- Segmenter: Abstract interface for segmenters
- DBSCANSegmenter: DBSCAN segmentation
- RegionGrowingSegmenter: Region growing segmentation
- ConnectedComponentsSegmenter: Connected components segmentation
- TreeSegmenter: Specialized tree segmentation
- BuildingSegmenter: Specialized building segmentation
- SegmentationManager: High-level manager with method selection

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import SegmentationResult, Segmenter
from .connected_components import ConnectedComponentsSegmenter
from .dbscan import DBSCANSegmenter
from .manager import SegmentationManager
from .region_growing import RegionGrowingSegmenter
from .specific import BuildingSegmenter, TreeSegmenter

__all__ = [
    "SegmentationResult",
    "Segmenter",
    "DBSCANSegmenter",
    "RegionGrowingSegmenter",
    "ConnectedComponentsSegmenter",
    "TreeSegmenter",
    "BuildingSegmenter",
    "SegmentationManager",
]
