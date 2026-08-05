from __future__ import annotations

from typing import Any

from topocore.dxf.constants import SCHEMA_VERSION
from topocore.features.models import Feature

DXFScalar = str | int | float | bool
_SCALAR_TYPES = (str, int, float, bool)
_KEY_CODE = 1000


class XDataEncoder:
    @staticmethod
    def encode(payload: dict[str, DXFScalar]) -> list[tuple[int, Any]]:
        tags: list[tuple[int, Any]] = []

        for key, value in payload.items():
            if not isinstance(value, _SCALAR_TYPES):
                raise TypeError(f"XDATA value for '{key}' must be one of {_SCALAR_TYPES}; got {type(value).__name__}.")

            tags.append((_KEY_CODE, key))

            if isinstance(value, bool):
                tags.append((1070, int(value)))
            elif isinstance(value, int):
                tags.append((1071, value))
            elif isinstance(value, float):
                tags.append((1040, value))
            else:
                tags.append((_KEY_CODE, value))

        return tags


class XDataDecoder:
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
    return [k for k, v in attributes.items() if not isinstance(v, _SCALAR_TYPES)]


def build_feature_xdata(feature: Feature) -> dict[str, DXFScalar]:
    payload: dict[str, DXFScalar] = {
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
        if isinstance(value, _SCALAR_TYPES):
            payload[key] = value

    return payload


__all__ = ["DXFScalar", "XDataEncoder", "XDataDecoder", "non_scalar_attribute_keys", "build_feature_xdata"]
