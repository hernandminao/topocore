from __future__ import annotations

import importlib.util
from types import ModuleType

from topocore.dxf.exceptions import DXFExportError

_NOT_INSTALLED_ERROR = (
    "ezdxf is not installed. Install it with `pip install topocore[dxf]` (or `pip install ezdxf`) to export DXF files."
)


def is_available() -> bool:
    return importlib.util.find_spec("ezdxf") is not None


def require_ezdxf() -> ModuleType:
    try:
        ezdxf = importlib.import_module("ezdxf")
    except ImportError as exc:
        raise DXFExportError(_NOT_INSTALLED_ERROR) from exc
    return ezdxf


__all__ = ["is_available", "require_ezdxf"]
