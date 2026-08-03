"""
topocore.analysis.statistics
============================

Statistics analysis sub-package.

Provides professional statistical analysis tools for:

- elevation;
- slope;
- triangulated surface area;
- point density;
- value distribution.

The package exposes both individual statistical processors
and the unified ``StatisticsAnalysis`` facade.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .area import AreaStatistics
from .density import DensityStatistics
from .distribution import DistributionStatistics
from .elevation import ElevationStatistics
from .manager import StatisticsAnalysis, StatisticsMethod
from .slope import SlopeStatistics

__all__ = [
    "AreaStatistics",
    "DensityStatistics",
    "DistributionStatistics",
    "ElevationStatistics",
    "SlopeStatistics",
    "StatisticsAnalysis",
    "StatisticsMethod",
]
