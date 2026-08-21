"""
Regression suite for topocore.features.catalogs.catalog_audit and
._validation -- PR19.

Verified the real, shipped catalog (ALL_CODES, 159 codes) passes
run_audit() with ZERO violations, and verified EACH of the 9
individual domain catalogs also passes independently. This alone is
a strong positive signal, but doesn't prove the checker actually
checks anything -- so each of the 5 violation categories was also
confirmed to fire correctly against deliberately-broken synthetic
definitions, ruling out a silent no-op checker. Also verified
_validation.py's GROUND/feature_type cross-checks and the
feature_type<->geometry_type compatibility check (_EXPECTED_GEOMETRY)
with real cases. No bugs found.
"""

from __future__ import annotations

import pytest

from topocore.features.catalogs import ALL_CODES
from topocore.features.catalogs._validation import (
    CatalogGeometryError,
    validate_definition_geometry,
)
from topocore.features.catalogs.cadastre import CADASTRE_CODES
from topocore.features.catalogs.catalog_audit import run_audit
from topocore.features.catalogs.control import CONTROL_CODES
from topocore.features.catalogs.default import DEFAULT_CODES
from topocore.features.catalogs.drainage import DRAINAGE_CODES
from topocore.features.catalogs.structures import STRUCTURE_CODES
from topocore.features.catalogs.terrain import TERRAIN_CODES
from topocore.features.catalogs.transportation import TRANSPORTATION_CODES
from topocore.features.catalogs.utilities import UTILITY_CODES
from topocore.features.catalogs.vegetation import VEGETATION_CODES
from topocore.features.feature_codes import FeatureCodeDefinition, FeatureGeometryType
from topocore.features.models import FeatureCategory, FeatureType


def _def(
    code: str,
    feature_type: FeatureType | None = FeatureType.TREE,
    category: FeatureCategory = FeatureCategory.VEGETATION,
    geometry_type: FeatureGeometryType = FeatureGeometryType.POINT,
    layer: str = "LAYER",
    closed: bool = False,
    aliases: tuple[str, ...] = (),
) -> FeatureCodeDefinition:
    return FeatureCodeDefinition(
        code=code,
        name=f"Def {code}",
        feature_type=feature_type,
        category=category,
        geometry_type=geometry_type,
        layer=layer,
        closed=closed,
        aliases=aliases,
    )


# ----------------------------------------------------------------------
# The real, shipped catalog.
# ----------------------------------------------------------------------


def test_real_combined_catalog_passes_audit() -> None:
    report = run_audit(ALL_CODES)
    assert report.passed
    assert report.violations == ()
    assert report.total_codes == len(ALL_CODES)


@pytest.mark.parametrize(
    "codes",
    [
        CADASTRE_CODES,
        CONTROL_CODES,
        DRAINAGE_CODES,
        STRUCTURE_CODES,
        TERRAIN_CODES,
        TRANSPORTATION_CODES,
        UTILITY_CODES,
        VEGETATION_CODES,
        DEFAULT_CODES,
    ],
    ids=[
        "cadastre",
        "control",
        "drainage",
        "structures",
        "terrain",
        "transportation",
        "utilities",
        "vegetation",
        "default",
    ],
)
def test_each_individual_catalog_passes_audit(
    codes: tuple[FeatureCodeDefinition, ...],
) -> None:
    report = run_audit(codes)
    assert report.passed, [(v.kind, v.code, v.message) for v in report.violations]


# ----------------------------------------------------------------------
# Each violation category, confirmed to genuinely fire (not a no-op).
# ----------------------------------------------------------------------


def test_duplicate_code_detected() -> None:
    d1 = _def("ARBOL", feature_type=FeatureType.TREE)
    d2 = _def("ARBOL", feature_type=FeatureType.SHRUB)  # same code, different content
    report = run_audit((d1, d2))
    assert any(v.kind == "duplicate_code" for v in report.violations)
    assert not report.passed


def test_duplicate_alias_detected() -> None:
    d1 = _def("MURO", aliases=("WALL",))
    d2 = _def("CERCA", aliases=("WALL",))  # same alias, different code
    report = run_audit((d1, d2))
    assert any(v.kind == "duplicate_alias" for v in report.violations)


def test_empty_layer_detected() -> None:
    report = run_audit((_def("VACIO", layer=""),))
    assert any(v.kind == "empty_layer" for v in report.violations)


def test_invalid_closed_detected() -> None:
    report = run_audit(
        (
            _def(
                "LINEA",
                feature_type=FeatureType.WALL,
                category=FeatureCategory.BUILDING,
                geometry_type=FeatureGeometryType.LINE,
                closed=True,
            ),
        )
    )
    assert any(v.kind == "invalid_closed" for v in report.violations)


def test_inconsistent_category_detected() -> None:
    d1 = _def("A", feature_type=FeatureType.TREE, category=FeatureCategory.VEGETATION)
    d2 = _def("B", feature_type=FeatureType.TREE, category=FeatureCategory.BUILDING)
    report = run_audit((d1, d2))
    assert any(v.kind == "inconsistent_category" for v in report.violations)


def test_clean_definitions_produce_zero_violations() -> None:
    d1 = _def("OK1")
    d2 = _def("OK2")
    report = run_audit((d1, d2))
    assert report.passed
    assert report.violations == ()


def test_coverage_percent_and_average_codes_per_type() -> None:
    report = run_audit(ALL_CODES)
    assert 0.0 <= report.coverage_percent <= 100.0
    assert report.average_codes_per_type > 0.0


def test_report_to_dict_and_str_do_not_raise() -> None:
    report = run_audit(ALL_CODES)
    as_dict = report.to_dict()
    assert as_dict["passed"] is True
    assert "PASS" in str(report)


# ----------------------------------------------------------------------
# _validation.py -- GROUND/feature_type and geometry compatibility.
# ----------------------------------------------------------------------


def test_ground_with_feature_type_rejected() -> None:
    definition = _def("G1", feature_type=FeatureType.TREE, geometry_type=FeatureGeometryType.GROUND)
    with pytest.raises(CatalogGeometryError):
        validate_definition_geometry(definition)


def test_ground_without_feature_type_accepted() -> None:
    definition = _def("G2", feature_type=None, geometry_type=FeatureGeometryType.GROUND)
    validate_definition_geometry(definition)  # must not raise


def test_non_ground_without_feature_type_rejected() -> None:
    definition = _def("P1", feature_type=None, geometry_type=FeatureGeometryType.POINT)
    with pytest.raises(CatalogGeometryError):
        validate_definition_geometry(definition)


def test_incompatible_feature_type_geometry_rejected() -> None:
    """TREE only allows POINT-family geometry per _EXPECTED_GEOMETRY -- LINE must be rejected."""
    definition = _def("T2", feature_type=FeatureType.TREE, geometry_type=FeatureGeometryType.LINE)
    with pytest.raises(CatalogGeometryError):
        validate_definition_geometry(definition)


def test_compatible_feature_type_geometry_accepted() -> None:
    definition = _def("T1", feature_type=FeatureType.TREE, geometry_type=FeatureGeometryType.POINT)
    validate_definition_geometry(definition)  # must not raise
