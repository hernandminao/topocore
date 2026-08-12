"""
topocore.dxf.xdata
=====================

Typed, symmetric XDATA encoding under APPID "TOPOCORE".

Supports two kinds of ``Feature.attributes`` values:

* A plain scalar (``str | int | float | bool``) -- encoded directly
  as its own typed group code (1000/1040/1070/1071).
* A ``tuple``/``list`` whose every element is itself a plain scalar
  -- encoded as a single compact-JSON string (group code 1000), so
  survey provenance like ``survey_point_ids`` survives the DXF
  round-trip losslessly and unambiguously (a plain ``"|"``/``","``
  join would break if a field ID itself contained that character;
  JSON doesn't have that problem).

Anything else (``dict``, nested collections, arbitrary objects) is
rejected -- ``DXFValidator``'s DXF002 check still fires for those,
so XDATA never becomes a dumping ground for arbitrary Python state.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import json
from typing import Any

from topocore.dxf.constants import SCHEMA_VERSION
from topocore.features.models import Feature

DXFScalar = str | int | float | bool
DXFEncodable = DXFScalar | tuple[DXFScalar, ...] | list[DXFScalar]

_SCALAR_TYPES = (str, int, float, bool)
_COLLECTION_TYPES = (tuple, list)
_KEY_CODE = 1000


def _is_scalar(value: object) -> bool:
    return isinstance(value, _SCALAR_TYPES)


def _is_scalar_collection(value: object) -> bool:
    return isinstance(value, _COLLECTION_TYPES) and all(_is_scalar(v) for v in value)


def is_xdata_encodable(value: object) -> bool:
    """Whether `value` can be represented in TopoCore XDATA (scalar, or a flat collection of scalars)."""
    return _is_scalar(value) or _is_scalar_collection(value)


class XDataEncoder:
    """Encodes a flat payload into ezdxf XDATA tags. See module docstring for the encoding scheme."""

    @staticmethod
    def encode(payload: dict[str, DXFEncodable]) -> list[tuple[int, Any]]:
        tags: list[tuple[int, Any]] = []

        for key, value in payload.items():
            if isinstance(value, _SCALAR_TYPES):
                encodable: DXFScalar = value
            elif isinstance(value, _COLLECTION_TYPES) and all(_is_scalar(v) for v in value):
                encodable = json.dumps(list(value), separators=(",", ":"))
            else:
                raise TypeError(
                    f"XDATA value for '{key}' must be a scalar ({_SCALAR_TYPES}) or a "
                    f"flat tuple/list of scalars; got {type(value).__name__}."
                )

            tags.append((_KEY_CODE, key))

            if isinstance(encodable, bool):
                tags.append((1070, int(encodable)))
            elif isinstance(encodable, int):
                tags.append((1071, encodable))
            elif isinstance(encodable, float):
                tags.append((1040, encodable))
            else:
                tags.append((_KEY_CODE, encodable))

        return tags


class XDataDecoder:
    """
    Symmetric counterpart to `XDataEncoder`. A strict contract: any
    structural violation raises `ValueError` rather than guessing.

    Values encoded from a scalar collection come back as their raw
    JSON string (e.g. ``'["W1","W2","W3"]'``) -- callers that know a
    given key holds a collection are expected to ``json.loads()`` it
    themselves; the decoder doesn't guess which strings are JSON.
    """

    @staticmethod
    def decode(tags: list[tuple[int, Any]]) -> dict[str, DXFScalar]:
        pairs = list(tags)

        if len(pairs) % 2 != 0:
            raise ValueError(f"Malformed TopoCore XDATA: expected an even number of key/value tags, got {len(pairs)}.")

        payload: dict[str, DXFScalar] = {}

        for i in range(0, len(pairs), 2):
            key_code, key = pairs[i]
            value_code, value = pairs[i + 1]

            if key_code != _KEY_CODE or not isinstance(key, str):
                raise ValueError(
                    f"Malformed TopoCore XDATA: expected a code-{_KEY_CODE} string key "
                    f"at position {i}, got (code={key_code}, value={value!r})."
                )

            match value_code:
                case 1000:
                    decoded: DXFScalar = str(value)
                case 1040:
                    decoded = float(value)
                case 1070:
                    decoded = bool(value)
                case 1071:
                    decoded = int(value)
                case _:
                    raise ValueError(
                        f"Malformed TopoCore XDATA: unrecognized value group code {value_code} for key '{key}'."
                    )

            payload[key] = decoded

        return payload


def non_scalar_attribute_keys(attributes: dict[str, Any]) -> list[str]:
    """Keys in `attributes` that `XDataEncoder` would reject (not scalar, not a flat scalar collection)."""
    return [k for k, v in attributes.items() if not is_xdata_encodable(v)]


def build_feature_xdata(feature: Feature) -> dict[str, DXFEncodable]:
    payload: dict[str, DXFEncodable] = {
        "schema_version": SCHEMA_VERSION,
        "feature_id": feature.feature_id if feature.feature_id is not None else -1,
        "feature_type": feature.feature_type.value,
        "category": feature.category.value,
        "confidence": feature.confidence,
    }

    if feature.metadata is not None:
        payload["detector"] = feature.metadata.detector
        payload["detector_version"] = feature.metadata.version

    for key, value in feature.attributes.items():
        if is_xdata_encodable(value):
            payload[key] = value

    return payload


__all__ = [
    "DXFEncodable",
    "DXFScalar",
    "XDataDecoder",
    "XDataEncoder",
    "build_feature_xdata",
    "is_xdata_encodable",
    "non_scalar_attribute_keys",
]
