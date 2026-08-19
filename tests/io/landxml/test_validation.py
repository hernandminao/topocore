"""
Tests for LandXMLValidator, used standalone (not only via
Reader/Writer), per the frozen PR18B contract.
"""

from __future__ import annotations

from xml.etree.ElementTree import fromstring

import pytest

from topocore.io.landxml.exceptions import LandXMLValidationError
from topocore.io.landxml.models import LandXMLDocument, NamedPointGroup, NamedSurface
from topocore.io.landxml.validation import LandXMLValidator
from topocore.survey.models import SurveyPoint, SurveyPointSet
from topocore.terrain.tin import TIN


def test_validate_xml_accepts_well_formed_subset() -> None:
    root = fromstring(
        """<LandXML>
            <Surfaces>
                <Surface name="S1">
                    <Definition surfType="TIN">
                        <Pnts>
                            <P id="1">0 0 0</P>
                            <P id="2">0 1 0</P>
                            <P id="3">1 0 0</P>
                        </Pnts>
                        <Faces><F>1 2 3</F></Faces>
                    </Definition>
                </Surface>
            </Surfaces>
        </LandXML>"""
    )

    LandXMLValidator().validate_xml(root)  # must not raise


def test_validate_xml_rejects_missing_surface_name() -> None:
    root = fromstring("<LandXML><Surfaces><Surface></Surface></Surfaces></LandXML>")

    with pytest.raises(LandXMLValidationError):
        LandXMLValidator().validate_xml(root)


def test_validate_xml_rejects_duplicated_surface_names() -> None:
    root = fromstring('<LandXML><Surfaces><Surface name="A"/><Surface name="A"/></Surfaces></LandXML>')

    with pytest.raises(LandXMLValidationError):
        LandXMLValidator().validate_xml(root)


def test_validate_xml_rejects_face_with_wrong_id_count() -> None:
    root = fromstring(
        """<LandXML>
            <Surfaces><Surface name="S1"><Definition surfType="TIN">
                <Pnts><P id="1">0 0 0</P><P id="2">0 1 0</P></Pnts>
                <Faces><F>1 2</F></Faces>
            </Definition></Surface></Surfaces>
        </LandXML>"""
    )

    with pytest.raises(LandXMLValidationError):
        LandXMLValidator().validate_xml(root)


def test_validate_xml_rejects_duplicated_cgpoint_names() -> None:
    root = fromstring(
        '<LandXML><CgPoints name="G"><CgPoint name="1">0 0</CgPoint>'
        '<CgPoint name="1">1 1</CgPoint></CgPoints></LandXML>'
    )

    with pytest.raises(LandXMLValidationError):
        LandXMLValidator().validate_xml(root)


def test_validate_document_rejects_duplicated_surface_names(
    two_triangle_square_tin: TIN,
) -> None:
    document = LandXMLDocument(
        surfaces=(
            NamedSurface(name="Dup", tin=two_triangle_square_tin),
            NamedSurface(name="Dup", tin=two_triangle_square_tin),
        )
    )

    with pytest.raises(LandXMLValidationError):
        LandXMLValidator().validate_document(document)


def test_validate_document_rejects_blank_point_group_name() -> None:
    group = NamedPointGroup(
        name="   ",
        points=SurveyPointSet(points=(SurveyPoint(id="1", x=0.0, y=0.0, z=0.0),)),
    )
    document = LandXMLDocument(point_groups=(group,))

    with pytest.raises(LandXMLValidationError):
        LandXMLValidator().validate_document(document)


def test_validate_document_rejects_non_finite_survey_point() -> None:
    group = NamedPointGroup(
        name="G",
        points=SurveyPointSet(points=(SurveyPoint(id="1", x=float("nan"), y=0.0, z=0.0),)),
    )
    document = LandXMLDocument(point_groups=(group,))

    with pytest.raises(LandXMLValidationError):
        LandXMLValidator().validate_document(document)


def test_validate_document_accepts_valid_document(two_triangle_square_tin: TIN) -> None:
    document = LandXMLDocument(
        surfaces=(NamedSurface(name="S1", tin=two_triangle_square_tin),),
        point_groups=(
            NamedPointGroup(
                name="G1",
                points=SurveyPointSet(points=(SurveyPoint(id="1", x=0.0, y=0.0, z=0.0),)),
            ),
        ),
    )

    LandXMLValidator().validate_document(document)  # must not raise
