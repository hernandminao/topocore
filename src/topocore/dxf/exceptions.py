from __future__ import annotations

from topocore.core.exceptions import TopoCoreError


class DXFError(TopoCoreError):
    """Base exception for all DXF export errors."""


class DXFValidationError(DXFError):
    """Raised by `DXFValidator` when `strict=True` and a feature has an ERROR-severity issue."""


class DXFGeometryError(DXFError):
    """Raised when `GeometryMapper` cannot represent a geometry safely."""


class DXFExportError(DXFError):
    """Raised for failures during the ezdxf write/save step, or a missing ezdxf install."""


__all__ = ["DXFError", "DXFValidationError", "DXFGeometryError", "DXFExportError"]
