"""
Regression suite for topocore.features.feature_codes -- PR19.

Verified alias resolution, case-insensitivity, deduplication of a
single definition registered under multiple keys, and the subtle
distinction between re-registering an identical definition (silent
success) vs. a genuinely conflicting one under the same key
(rejected, unless overwrite=True). No bugs found.
"""

from __future__ import annotations

import pytest

from topocore.features.feature_codes import (
    FeatureCodeDefinition,
    FeatureCodeRegistry,
    FeatureGeometryType,
)
from topocore.features.models import FeatureCategory, FeatureType


def _make_def(
    code: str,
    aliases: tuple[str, ...] = (),
    feature_type: FeatureType = FeatureType.TREE,
) -> FeatureCodeDefinition:
    return FeatureCodeDefinition(
        code=code,
        name=f"Definition for {code}",
        feature_type=feature_type,
        category=FeatureCategory.VEGETATION,
        geometry_type=FeatureGeometryType.POINT,
        layer="TEST",
        aliases=aliases,
    )


def test_alias_and_case_insensitive_lookup() -> None:
    registry = FeatureCodeRegistry()
    definition = _make_def("ARBOL", aliases=("TREE", "TREE1"))
    registry.register(definition)

    assert registry.get("ARBOL") is definition
    assert registry.get("arbol") is definition
    assert registry.get("TREE") is definition
    assert registry.get("tree1") is definition


def test_definitions_deduplicated_across_keys() -> None:
    registry = FeatureCodeRegistry()
    registry.register(_make_def("ARBOL", aliases=("TREE", "TREE1")))
    assert len(registry) == 1


def test_reregistering_identical_definition_is_a_silent_success() -> None:
    registry = FeatureCodeRegistry()
    definition = _make_def("ARBOL")
    registry.register(definition)
    registry.register(definition)  # must not raise


def test_conflicting_definition_under_same_code_rejected() -> None:
    registry = FeatureCodeRegistry()
    registry.register(_make_def("ARBOL", feature_type=FeatureType.TREE))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_make_def("ARBOL", feature_type=FeatureType.SHRUB))


def test_overwrite_true_bypasses_conflict() -> None:
    registry = FeatureCodeRegistry()
    registry.register(_make_def("ARBOL", feature_type=FeatureType.TREE))
    replacement = _make_def("ARBOL", feature_type=FeatureType.SHRUB)
    registry.register(replacement, overwrite=True)
    assert registry.get("ARBOL") is replacement


def test_alias_collision_with_different_base_code_rejected() -> None:
    registry = FeatureCodeRegistry()
    registry.register(_make_def("ARBOL", aliases=("TREE",)))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_make_def("MURO", aliases=("TREE",)))


def test_unregistered_code_returns_none() -> None:
    registry = FeatureCodeRegistry()
    assert registry.get("NONEXISTENT") is None
