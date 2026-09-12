"""
topocore.geodesy.georeferencing
==================================

Control-point-based georeferencing: estimating a 3D Helmert
similarity transformation from known correspondences (local/arbitrary
source coordinates paired with their known target coordinates), as
opposed to `topocore.geodesy.CoordinateTransformer`, which applies an
already-known transformation between two already-identified CRS.

This first slice is deliberately narrow -- the pure mathematical
core only: `ControlPoint`, `fit_helmert_3d()`, `apply_helmert_3d()`.
No `Georeferencing`/`GeoreferencingOptions` orchestration, no
acceptance validation, no `SurveyPointSet`/`target_crs` integration
yet -- those build on top of this core in a later phase.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geodesy.georeferencing.apply import (
    apply_georeferencing_to_feature_collection,
    apply_georeferencing_to_point_cloud,
    apply_georeferencing_to_survey,
    apply_helmert_3d_array,
)
from topocore.geodesy.georeferencing.control_point import ControlPoint
from topocore.geodesy.georeferencing.factory import fit_georeferencing
from topocore.geodesy.georeferencing.helmert_2d import fit_helmert_2d
from topocore.geodesy.georeferencing.helmert_fit import (
    HelmertFitResult,
    Residual,
    apply_helmert_3d,
    fit_helmert_3d,
)
from topocore.geodesy.georeferencing.options import GeoreferencingOptions
from topocore.geodesy.georeferencing.result import (
    GeoreferencingResult,
    GeoreferencingStrategy,
)
from topocore.geodesy.georeferencing.single_control import fit_translation_only

__all__ = [
    "ControlPoint",
    "GeoreferencingOptions",
    "GeoreferencingResult",
    "GeoreferencingStrategy",
    "HelmertFitResult",
    "Residual",
    "apply_georeferencing_to_feature_collection",
    "apply_georeferencing_to_point_cloud",
    "apply_georeferencing_to_survey",
    "apply_helmert_3d",
    "apply_helmert_3d_array",
    "fit_georeferencing",
    "fit_helmert_2d",
    "fit_helmert_3d",
    "fit_translation_only",
]
