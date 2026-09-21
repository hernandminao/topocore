"""
Regression tests for topocore.analysis.profile.dxf_export -- V1
scope confirmed explicitly: real terrain only, no design surface,
no cut/fill area, no road template (curbs, ditches, berms).
"""

from __future__ import annotations

import ezdxf
import pytest
from topocore.analysis.profile.dxf_export import (
    CrossSectionDXFExportError,
    export_cross_sections_dxf,
)
from topocore.analysis.types import ProfilePoint, ProfileResult, ProfileType


def _section(station: float, offsets_and_z: list[tuple[float, float]]) -> ProfileResult:
    points = [ProfilePoint(station=station, x=offset, y=0.0, z=z, offset=offset) for offset, z in offsets_and_z]
    return ProfileResult(points=points, profile_type=ProfileType.TRANSVERSAL)


def _polylines(path: str) -> list:
    doc = ezdxf.readfile(path)
    return [e for e in doc.modelspace() if e.dxftype() == "LWPOLYLINE"]


def _texts(path: str) -> list:
    doc = ezdxf.readfile(path)
    return [e for e in doc.modelspace() if e.dxftype() == "TEXT"]


def test_draws_one_polyline_per_section_with_offset_as_x_and_elevation_as_y(tmp_path) -> None:
    section = _section(0.0, [(-5.0, 10.0), (0.0, 10.5), (5.0, 10.9)])

    output = tmp_path / "sections.dxf"
    export_cross_sections_dxf([section], output)

    polylines = _polylines(str(output))
    assert len(polylines) == 1
    points = list(polylines[0].get_points("xy"))
    # offset minimo (-5.0) queda en x=0 -- convencion local por seccion
    assert points[0] == pytest.approx((0.0, 10.0))
    assert points[1] == pytest.approx((5.0, 10.5))
    assert points[2] == pytest.approx((10.0, 10.9))


def test_multiple_sections_are_laid_out_side_by_side_without_overlapping(tmp_path) -> None:
    first = _section(0.0, [(-8.0, 10.0), (8.0, 10.9)])  # ancho real: 16.0
    second = _section(20.0, [(-8.0, 12.3), (8.0, 14.7)])

    output = tmp_path / "sections.dxf"
    export_cross_sections_dxf([first, second], output, horizontal_spacing=40.0)

    polylines = _polylines(str(output))
    first_points = list(polylines[0].get_points("xy"))
    second_points = list(polylines[1].get_points("xy"))

    assert first_points[0][0] == pytest.approx(0.0)
    # 16.0 (ancho real de la primera) + 40.0 (espaciado) = 56.0
    assert second_points[0][0] == pytest.approx(56.0)


def test_station_label_uses_the_real_station_from_profilepoint(tmp_path) -> None:
    section = _section(125.40, [(-5.0, 10.0), (5.0, 10.5)])

    output = tmp_path / "sections.dxf"
    export_cross_sections_dxf([section], output)

    texts = _texts(str(output))
    etiquetas_estacion = [t.dxf.text for t in texts if t.dxf.text.startswith("K")]
    assert etiquetas_estacion == ["K125.40"]


def test_a_section_with_fewer_than_two_points_is_rejected() -> None:
    section = _section(0.0, [(-5.0, 10.0)])

    with pytest.raises(CrossSectionDXFExportError, match="fewer than 2 points"):
        export_cross_sections_dxf([section], "/tmp/does_not_matter.dxf")


def test_nonpositive_horizontal_spacing_is_rejected() -> None:
    section = _section(0.0, [(-5.0, 10.0), (5.0, 10.5)])

    with pytest.raises(CrossSectionDXFExportError, match="horizontal_spacing"):
        export_cross_sections_dxf([section], "/tmp/does_not_matter.dxf", horizontal_spacing=0.0)


def test_nonpositive_text_height_is_rejected() -> None:
    section = _section(0.0, [(-5.0, 10.0), (5.0, 10.5)])

    with pytest.raises(CrossSectionDXFExportError, match="text_height"):
        export_cross_sections_dxf([section], "/tmp/does_not_matter.dxf", text_height=0.0)


def test_endpoint_elevations_are_labeled_with_real_values(tmp_path) -> None:
    section = _section(0.0, [(-5.0, 10.25), (5.0, 12.75)])

    output = tmp_path / "sections.dxf"
    export_cross_sections_dxf([section], output)

    valores = {t.dxf.text for t in _texts(str(output))}
    assert "10.25" in valores
    assert "12.75" in valores


def test_lowest_point_is_marked_and_labeled(tmp_path) -> None:
    section = _section(0.0, [(-5.0, 10.0), (0.0, 8.5), (5.0, 11.0)])

    output = tmp_path / "sections.dxf"
    export_cross_sections_dxf([section], output)

    doc = ezdxf.readfile(str(output))
    circulos = [e for e in doc.modelspace() if e.dxftype() == "CIRCLE"]
    assert len(circulos) == 1
    assert circulos[0].dxf.center[1] == pytest.approx(8.5)
    assert "8.50" in {t.dxf.text for t in _texts(str(output))}


def test_elevation_axis_draws_ticks_at_the_given_interval(tmp_path) -> None:
    section = _section(0.0, [(-5.0, 10.0), (5.0, 13.4)])

    output = tmp_path / "sections.dxf"
    export_cross_sections_dxf([section], output, elevation_axis_interval=1.0)

    valores = {t.dxf.text for t in _texts(str(output))}
    # 10.00, 11.00, 12.00, 13.00 deben aparecer como marcas de la regla
    for cota in ("10.00", "11.00", "12.00", "13.00"):
        assert cota in valores


def test_all_three_elevation_labels_can_be_disabled(tmp_path) -> None:
    section = _section(0.0, [(-5.0, 10.0), (0.0, 8.5), (5.0, 11.0)])

    output = tmp_path / "sections.dxf"
    export_cross_sections_dxf(
        [section],
        output,
        show_endpoint_elevations=False,
        show_lowest_point_elevation=False,
        show_elevation_axis=False,
    )

    textos = [t.dxf.text for t in _texts(str(output))]
    assert textos == ["K0.00"]  # solo la etiqueta de estacion queda


def test_custom_layer_name_is_applied_to_geometry_and_labels(tmp_path) -> None:
    section = _section(0.0, [(-5.0, 10.0), (5.0, 10.5)])

    output = tmp_path / "sections.dxf"
    export_cross_sections_dxf([section], output, layer="MI_CAPA")

    assert _polylines(str(output))[0].dxf.layer == "MI_CAPA"
    assert _texts(str(output))[0].dxf.layer == "MI_CAPA"


def test_returns_the_path_it_was_given(tmp_path) -> None:
    section = _section(0.0, [(-5.0, 10.0), (5.0, 10.5)])

    output = tmp_path / "sections.dxf"
    result = export_cross_sections_dxf([section], output)

    assert result == output
