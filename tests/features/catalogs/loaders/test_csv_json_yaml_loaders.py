"""
Regression suite for topocore.features.catalogs.loaders (csv_loader,
json_loader, yaml_loader) -- PR19.

Verified with real files (not mocks) for all 3 formats: valid round
trips (including Spanish boolean values "sí"/"no" for CSV, and
semicolon-separated vs. list-based aliases), missing required
columns (CSV), and the decisive strict-boolean-typing check: a JSON
"closed": "false" (string) is correctly REJECTED, not silently
coerced to Python's bool("false") == True -- confirmed this would
otherwise be a genuine catalog-authoring trap. Also verified multiple
errors in one file are all reported together, not just the first.
No bugs found.
"""

from __future__ import annotations

import json

import pytest

from topocore.features.catalogs.loaders.base import ExternalCatalogError
from topocore.features.catalogs.loaders.csv_loader import load_csv
from topocore.features.catalogs.loaders.json_loader import load_json
from topocore.features.catalogs.loaders.yaml_loader import load_yaml

# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------


def test_csv_valid_round_trip_with_spanish_boolean_and_semicolon_aliases(
    tmp_path: object,
) -> None:
    path = tmp_path / "catalog.csv"  # type: ignore[operator]
    path.write_text(
        "code,name,geometry_type,feature_type,category,layer,closed,aliases\n"
        "EDIF,Edificio,polygon,building,building,BUILDINGS,sí,BLDG;STRUCT\n",
        encoding="utf-8-sig",
    )

    result = load_csv(str(path))

    assert len(result) == 1
    assert result[0].closed is True
    assert result[0].aliases == ("BLDG", "STRUCT")


def test_csv_missing_required_column_rejected(tmp_path: object) -> None:
    path = tmp_path / "bad.csv"  # type: ignore[operator]
    path.write_text("code,name,geometry_type\nX,Y,point\n", encoding="utf-8-sig")

    with pytest.raises(ValueError, match="missing required column"):
        load_csv(str(path))


def test_csv_unrecognized_boolean_value_rejected(tmp_path: object) -> None:
    path = tmp_path / "badbool.csv"  # type: ignore[operator]
    path.write_text(
        "code,name,geometry_type,feature_type,category,layer,closed\n"
        "X,Y,polygon,building,building,L,verdadero\n",  # not in the recognized set
        encoding="utf-8-sig",
    )

    with pytest.raises(ExternalCatalogError):
        load_csv(str(path))


def test_csv_empty_closed_cell_defaults_to_false(tmp_path: object) -> None:
    path = tmp_path / "empty_closed.csv"  # type: ignore[operator]
    path.write_text(
        "code,name,geometry_type,feature_type,category,layer,closed\nX,Y,point,tree,vegetation,L,\n",
        encoding="utf-8-sig",
    )

    result = load_csv(str(path))
    assert result[0].closed is False


# ----------------------------------------------------------------------
# JSON -- the decisive strict-boolean-typing check.
# ----------------------------------------------------------------------


def test_json_string_false_rejected_not_coerced(tmp_path: object) -> None:
    """
    The decisive check: Python's bool("false") is True. This loader
    must reject a string "closed" value outright rather than
    silently coercing it to the OPPOSITE of what it looks like.
    """
    path = tmp_path / "bad.json"  # type: ignore[operator]
    path.write_text(
        json.dumps(
            {
                "codes": [
                    {
                        "code": "X",
                        "name": "Y",
                        "geometry_type": "point",
                        "feature_type": "tree",
                        "category": "vegetation",
                        "layer": "L",
                        "closed": "false",
                    }
                ]
            }
        )
    )

    with pytest.raises(ExternalCatalogError):
        load_json(str(path))


def test_json_valid_round_trip(tmp_path: object) -> None:
    path = tmp_path / "catalog.json"  # type: ignore[operator]
    path.write_text(
        json.dumps(
            {
                "codes": [
                    {
                        "code": "MURO",
                        "name": "Muro",
                        "geometry_type": "line",
                        "feature_type": "wall",
                        "category": "building",
                        "layer": "WALLS",
                        "aliases": ["WALL", "PARED"],
                    }
                ]
            }
        )
    )

    result = load_json(str(path))
    assert len(result) == 1
    assert result[0].aliases == ("WALL", "PARED")


def test_json_multiple_errors_reported_together(tmp_path: object) -> None:
    path = tmp_path / "multi.json"  # type: ignore[operator]
    path.write_text(
        json.dumps(
            {
                "codes": [
                    {
                        "code": "",
                        "name": "Y",
                        "geometry_type": "point",
                        "feature_type": "tree",
                        "category": "vegetation",
                        "layer": "L",
                    },
                    {
                        "code": "X2",
                        "name": "Y2",
                        "geometry_type": "bogus",
                        "feature_type": "tree",
                        "category": "vegetation",
                        "layer": "L",
                    },
                    {
                        "code": "X3",
                        "name": "Y3",
                        "geometry_type": "point",
                        "feature_type": "bogus_type",
                        "category": "vegetation",
                        "layer": "L",
                    },
                ]
            }
        )
    )

    with pytest.raises(ExternalCatalogError) as exc_info:
        load_json(str(path))

    assert len(exc_info.value.issues) == 3


def test_json_root_must_be_object(tmp_path: object) -> None:
    path = tmp_path / "notobject.json"  # type: ignore[operator]
    path.write_text(json.dumps([1, 2, 3]))

    with pytest.raises(TypeError):
        load_json(str(path))


def test_json_unknown_category_rejected(tmp_path: object) -> None:
    path = tmp_path / "badcategory.json"  # type: ignore[operator]
    path.write_text(
        json.dumps(
            {
                "codes": [
                    {
                        "code": "X",
                        "name": "Y",
                        "geometry_type": "point",
                        "feature_type": "tree",
                        "category": "not_a_real_category",
                        "layer": "L",
                    }
                ]
            }
        )
    )

    with pytest.raises(ExternalCatalogError):
        load_json(str(path))


# ----------------------------------------------------------------------
# YAML
# ----------------------------------------------------------------------


def test_yaml_valid_round_trip_with_list_aliases(tmp_path: object) -> None:
    path = tmp_path / "catalog.yaml"  # type: ignore[operator]
    path.write_text(
        "codes:\n"
        "  - code: MURO\n"
        "    name: Muro\n"
        "    geometry_type: line\n"
        "    feature_type: wall\n"
        "    category: building\n"
        "    layer: WALLS\n"
        "    aliases: [WALL, PARED]\n"
    )

    result = load_yaml(str(path))
    assert len(result) == 1
    assert result[0].aliases == ("WALL", "PARED")


def test_yaml_string_false_rejected_not_coerced(tmp_path: object) -> None:
    path = tmp_path / "bad.yaml"  # type: ignore[operator]
    path.write_text(
        "codes:\n"
        "  - code: X\n"
        "    name: Y\n"
        "    geometry_type: point\n"
        "    feature_type: tree\n"
        "    category: vegetation\n"
        "    layer: L\n"
        '    closed: "false"\n'  # quoted, so YAML parses it as a string, not a bool
    )

    with pytest.raises(ExternalCatalogError):
        load_yaml(str(path))


def test_yaml_root_must_be_mapping(tmp_path: object) -> None:
    path = tmp_path / "notmapping.yaml"  # type: ignore[operator]
    path.write_text("- 1\n- 2\n- 3\n")

    with pytest.raises(TypeError):
        load_yaml(str(path))
