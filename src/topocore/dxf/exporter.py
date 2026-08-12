from __future__ import annotations

import importlib
from pathlib import Path

from topocore.dxf._ezdxf_compat import is_available, require_ezdxf
from topocore.dxf.constants import APPID
from topocore.dxf.entities import write_entity
from topocore.dxf.exceptions import DXFExportError, DXFGeometryError, DXFValidationError
from topocore.dxf.layers import LAYER_STYLES, contour_layer_name, layer_for
from topocore.dxf.mapping import DXFRepresentation, GeometryMapper
from topocore.dxf.models import DrawingUnits, ExportContext
from topocore.dxf.report import DXFExportReport, _ReportBuilder
from topocore.dxf.validation import DXFValidator, ValidationSeverity
from topocore.dxf.xdata import build_feature_xdata
from topocore.features.models import Feature, FeatureCollection, FeatureType

_INSUNITS_ATTR: dict[DrawingUnits, str] = {
    DrawingUnits.METERS: "M",
    DrawingUnits.MILLIMETERS: "MM",
    DrawingUnits.FEET: "FT",
}


class DXFExporter:
    __slots__ = ("_context", "_mapper", "_validator")

    def __init__(self, context: ExportContext | None = None) -> None:
        if not is_available():
            raise DXFExportError(
                "ezdxf is not installed. Install it with `pip install topocore[dxf]` "
                "(or `pip install ezdxf`) to use DXFExporter."
            )

        self._context = context or ExportContext()
        options = self._context.options
        self._validator = DXFValidator()
        self._mapper = GeometryMapper(options.tolerance, non_planar_polygon_mode=options.non_planar_polygon_mode)

    @staticmethod
    def is_available() -> bool:
        return is_available()

    def export(self, collection: FeatureCollection, path: str | Path) -> DXFExportReport:
        ezdxf = require_ezdxf()

        ezdxf_units = importlib.import_module("ezdxf.units")

        options = self._context.options
        report = _ReportBuilder()

        doc = ezdxf.new(options.dxf_version, setup=True)
        doc.appids.new(APPID)
        doc.units = getattr(ezdxf_units, _INSUNITS_ATTR[options.units])
        self._setup_layers(doc)

        if self._context.crs:
            doc.header.custom_vars.append("TopoCore CRS", self._context.crs)
        doc.header.custom_vars.append("TopoCore Units", options.units.value)

        msp = doc.modelspace()

        for feature in collection:
            report.feature_count += 1
            report.features_by_type[feature.feature_type] += 1

            issues = self._validator.validate(feature)
            for issue in issues:
                report.warnings.append(f"[{issue.code}] feature {issue.feature_id}: {issue.message}")

            errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
            if errors:
                if options.strict:
                    raise DXFValidationError(
                        f"Feature {feature.feature_id} failed DXF validation: "
                        f"{'; '.join(f'[{i.code}] {i.message}' for i in errors)}"
                    )
                report.skipped_features += 1
                continue

            try:
                decision = self._mapper.decide(feature.geometry)
                layer = self._resolve_layer(feature, options.index_contour_every)
                xdata = build_feature_xdata(feature)
                entities = write_entity(msp, feature.geometry, decision, layer, xdata)

            except (DXFGeometryError, DXFExportError) as exc:
                # Only TopoCore's own, well-defined DXF failure modes
                # are catchable here. Anything else (a genuine bug)
                # propagates even in non-strict mode -- strict=False
                # skips bad *data*, it never turns a real bug into a
                # silent skip.
                if options.strict:
                    raise
                report.skipped_features += 1
                report.warnings.append(f"feature {feature.feature_id} skipped: {exc}")
                continue

            report.entity_count += len(entities)
            self._tally(report, decision.representation, len(entities))

        try:
            doc.saveas(str(path))
        except (OSError, ezdxf.DXFError) as exc:
            raise DXFExportError(f"Failed to save DXF file to '{path}': {exc}") from exc

        report.output_path = Path(path)
        report.dxf_version = options.dxf_version
        report.units = options.units
        report.layer_count = len(LAYER_STYLES)

        return report.finalize()

    def _setup_layers(self, doc: object) -> None:
        for style in LAYER_STYLES.values():
            layer = doc.layers.new(style.name)  # type: ignore[attr-defined]
            layer.color = style.color
            layer.dxf.linetype = style.linetype
            if style.lineweight is not None:
                layer.dxf.lineweight = style.lineweight

    def _resolve_layer(self, feature: Feature, index_contour_every: int) -> str:
        """
        Layer resolution precedence:

        1. ``feature.attributes["cad_layer"]``, if present -- an
           explicit layer the source of the feature already knows
           (survey field codes via `FeatureCodeDefinition.layer`).
           Never guessed from `FeatureType` when the data itself
           already declares it.
        2. Contour MAJOR/MINOR/neutral resolution, for
           `FeatureType.CONTOUR` specifically (PR15-only; contours
           never carry `cad_layer`).
        3. `LAYER_BY_FEATURE_TYPE`, the PR15-detector fallback table --
           only reached for features that declare neither.
        """
        cad_layer = feature.attributes.get("cad_layer")
        if isinstance(cad_layer, str) and cad_layer:
            return cad_layer

        if feature.feature_type == FeatureType.CONTOUR:
            extra = feature.metadata.extra if feature.metadata else {}
            return contour_layer_name(
                elevation=feature.attributes.get("elevation"),
                base=extra.get("base"),
                interval=extra.get("interval"),
                every=index_contour_every,
            )
        return layer_for(feature.feature_type)

    @staticmethod
    def _tally(report: _ReportBuilder, representation: DXFRepresentation, count: int) -> None:
        if representation == DXFRepresentation.POINT:
            report.point_count += count
        elif representation == DXFRepresentation.LWPOLYLINE:
            report.lwpolyline_count += count
        elif representation == DXFRepresentation.POLYLINE3D:
            report.polyline3d_count += count
        elif representation == DXFRepresentation.FACE3D:
            report.face3d_count += count


__all__ = ["DXFExporter"]
