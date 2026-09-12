"""
topocore.quality.pdf_renderer -- PROPUESTA, actualizado al contrato
del Punto 2. Único módulo que importa reportlab. Toma un QualityResult
ya calculado (sin volver a calcular nada) y lo presenta como PDF --
el extra que quedaría detrás de `pip install topocore[quality-report]`.
"""

from __future__ import annotations

from pathlib import Path

from topocore.quality.models import QualityResult


class PDFRendererUnavailableError(Exception):
    """reportlab no está instalado -- ver el mensaje para el extra a instalar."""


class PDFRenderError(Exception):
    """La escritura del PDF falló (ruta inválida, permisos, disco, etc.)."""


def render_pdf(result: QualityResult, output_path: str | Path) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise PDFRendererUnavailableError(
            "Rendering a QualityResult to PDF requires reportlab, which is not installed. "
            "Install it with `pip install topocore[quality-report]`."
        ) from exc

    output_path = Path(output_path)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Reporte de Calidad -- {result.project_name}", styles["Title"]), Spacer(1, 12)]

    ref = result.spatial_reference
    if ref is not None:
        story.append(Paragraph("Referencia espacial", styles["Heading2"]))
        story.append(Paragraph(f"CRS: {ref.crs or 'no especificado'}", styles["Normal"]))
        story.append(Spacer(1, 12))

    acc = result.accuracy
    if acc is not None:
        story.append(Paragraph("Exactitud posicional (indicador, no certificación)", styles["Heading2"]))
        tabla = Table(
            [
                ["Componente", f"Valor ({acc.units})"],
                ["Horizontal", f"{acc.rmse_horizontal:.4f}"],
                ["Vertical", f"{acc.rmse_vertical:.4f}"],
                ["Total", f"{acc.rmse_total:.4f}"],
                ["N. de puntos de control", str(acc.control_point_count)],
            ]
        )
        tabla.setStyle(
            TableStyle(
                [("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]
            )
        )
        story.append(tabla)
        story.append(Spacer(1, 12))

    density = result.density
    if density is not None:
        story.append(Paragraph("Densidad de puntos (indicador de adquisición)", styles["Heading2"]))
        story.append(
            Paragraph(
                f"Media: {density.mean:.2f} {density.units} "
                f"(mín: {density.minimum:.2f}, máx: {density.maximum:.2f}, "
                f"resolución de celda: {density.resolution} m)",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

    coverage = result.coverage
    if coverage is not None:
        story.append(Paragraph("Cobertura espacial (depende de cell_size, no es absoluta)", styles["Heading2"]))
        story.append(
            Paragraph(
                f"{coverage.occupied_cells:,} de {coverage.total_cells:,} celdas de "
                f"{coverage.cell_size}x{coverage.cell_size} m con al menos "
                f"{coverage.min_points_per_cell} punto(s) ({coverage.coverage_percentage:.1f}%)",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

    classification = result.classification
    if classification is not None:
        story.append(Paragraph("Clasificación (indicador descriptivo)", styles["Heading2"]))
        if classification.code_provided:
            story.append(
                Paragraph(
                    f"Código {classification.matched_code}: {classification.matched_points:,} de "
                    f"{classification.total_points:,} puntos ({classification.ground_percentage:.1f}%)",
                    styles["Normal"],
                )
            )
        else:
            story.append(
                Paragraph(
                    f"{classification.total_points:,} puntos disponibles -- ningún código especificado (N/A)",
                    styles["Normal"],
                )
            )
        story.append(Spacer(1, 12))

    if result.diagnostics:
        story.append(Paragraph("Diagnósticos", styles["Heading2"]))
        for d in result.diagnostics:
            story.append(Paragraph(f"[{d.severity.value.upper()}] {d.code}: {d.message}", styles["Normal"]))

    try:
        SimpleDocTemplate(str(output_path), pagesize=letter).build(story)
    except OSError as exc:
        raise PDFRenderError(f"Could not write PDF to '{output_path}': {exc}") from exc

    return output_path
