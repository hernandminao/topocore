"""
topocore.terrain
================

Terrain modeling package.

This package provides the infrastructure for building and analysing
terrain models from georeferenced point clouds.

Implemented components
----------------------

* TIN
* DTM
* Grid
* Raster
* Terrain interpolation
* Contours
* Slope / Aspect / Hillshade
* Filters (Laplacian smoothing, spike detection)
* Breaklines (constrained triangulation)
* Terrain models

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .aspect import AspectCalculator, triangle_aspect
from .base import (
    BaseDTM,
    BaseInterpolator,
    BaseTIN,
)
from .breaklines import BreaklineEnforcer, apply_breaklines
from .constants import (
    DEFAULT_CONTOUR_INTERVAL,
    DEFAULT_GRID_ROTATION,
    EPSILON,
)
from .contours import ContourGenerator
from .dtm import DTM
from .enums import BreaklineType
from .exceptions import (
    BreaklineError,
    ContourError,
    InterpolationError,
    TerrainError,
    TerrainValidationError,
    TriangulationError,
)
from .filters import (
    LaplacianSmoother,
    SpikeDetector,
    detect_spikes,
    laplacian_smooth,
    remove_spikes,
)
from .grid import Grid
from .hillshade import DEFAULT_ALTITUDE, DEFAULT_AZIMUTH, HillshadeCalculator, triangle_hillshade
from .interpolation import (
    BarycentricInterpolator,
    IDWInterpolator,
    LinearInterpolator,
    NearestInterpolator,
)
from .models import (
    Breakline,
    ContourLine,
    Edge,
    GridDefinition,
    Triangle,
)
from .raster import Raster
from .slope import SlopeCalculator, triangle_slope
from .tin import TIN
from .types import (
    Aspect,
    Elevation,
    Interval,
    Resolution,
    Slope,
)
from .validation import (
    validate_breakline,
    validate_grid_definition,
    validate_interval,
    validate_points,
    validate_resolution,
    validate_tin,
    validate_triangle,
)

__all__ = [
    # Base
    "BaseTIN",
    "BaseDTM",
    "BaseInterpolator",
    # Terrain
    "TIN",
    "DTM",
    # Raster
    "Grid",
    "Raster",
    # Interpolation
    "LinearInterpolator",
    "BarycentricInterpolator",
    "NearestInterpolator",
    "IDWInterpolator",
    # Contours
    "ContourGenerator",
    # Slope / Aspect / Hillshade
    "SlopeCalculator",
    "triangle_slope",
    "AspectCalculator",
    "triangle_aspect",
    "HillshadeCalculator",
    "triangle_hillshade",
    "DEFAULT_AZIMUTH",
    "DEFAULT_ALTITUDE",
    # Filters
    "LaplacianSmoother",
    "laplacian_smooth",
    "SpikeDetector",
    "detect_spikes",
    "remove_spikes",
    # Breaklines
    "BreaklineEnforcer",
    "apply_breaklines",
    # Models
    "Edge",
    "Triangle",
    "Breakline",
    "ContourLine",
    "GridDefinition",
    # Enums
    "BreaklineType",
    # Types
    "Elevation",
    "Resolution",
    "Interval",
    "Slope",
    "Aspect",
    # Constants
    "EPSILON",
    "DEFAULT_GRID_ROTATION",
    "DEFAULT_CONTOUR_INTERVAL",
    # Validation
    "validate_resolution",
    "validate_interval",
    "validate_points",
    "validate_triangle",
    "validate_breakline",
    "validate_grid_definition",
    "validate_tin",
    # Exceptions
    "TerrainError",
    "TerrainValidationError",
    "TriangulationError",
    "InterpolationError",
    "ContourError",
    "BreaklineError",
]
