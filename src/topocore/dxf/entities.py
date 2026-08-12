from __future__ import annotations

from typing import Any

from topocore.dxf._ezdxf_compat import require_ezdxf
from topocore.dxf.constants import APPID
from topocore.dxf.exceptions import DXFExportError, DXFGeometryError
from topocore.dxf.mapping import DXFRepresentation, MappingDecision
from topocore.dxf.xdata import DXFEncodable, XDataEncoder
from topocore.features.models import FeatureGeometry


def _pt(v: Any) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def write_entity(
    msp: Any,
    geometry: FeatureGeometry,
    decision: MappingDecision,
    layer: str,
    xdata: dict[str, DXFEncodable],
) -> list[Any]:
    ezdxf = require_ezdxf()
    entities: list[Any] = []

    try:
        if decision.representation == DXFRepresentation.POINT:
            p = _pt(geometry.vertices[0])
            entities.append(msp.add_point(p, dxfattribs={"layer": layer}))

        elif decision.representation == DXFRepresentation.LWPOLYLINE:
            if decision.elevation is None:
                raise DXFGeometryError("LWPOLYLINE representation requires a constant elevation.")
            points_2d = [(float(v[0]), float(v[1])) for v in geometry.vertices]
            entities.append(
                msp.add_lwpolyline(
                    points_2d,
                    close=geometry.closed,
                    dxfattribs={"layer": layer, "elevation": float(decision.elevation)},
                )
            )

        elif decision.representation == DXFRepresentation.POLYLINE3D:
            points_3d = [_pt(v) for v in geometry.vertices]
            entity = msp.add_polyline3d(points_3d, dxfattribs={"layer": layer})
            if geometry.closed:
                entity.close(True)
            entities.append(entity)

        elif decision.representation == DXFRepresentation.FACE3D:
            assert geometry.faces is not None
            for face in geometry.faces:
                tri = geometry.vertices[face]
                entities.append(
                    msp.add_3dface(
                        [_pt(tri[0]), _pt(tri[1]), _pt(tri[2]), _pt(tri[2])],
                        dxfattribs={"layer": layer},
                    )
                )

        else:
            raise NotImplementedError(f"No entity writer for {decision.representation}.")

        encoded = XDataEncoder.encode(xdata)
        for entity in entities:
            entity.set_xdata(APPID, encoded)

    except ezdxf.DXFError as exc:
        raise DXFExportError(f"ezdxf rejected entity write: {exc}") from exc

    return entities


__all__ = ["write_entity"]
