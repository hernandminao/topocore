"""
Integration test against horizontal alignment geometry extracted
from a genuine, unmodified LandXML file exported by Autodesk Civil
3D 2007 (GSG_features_alignments.xml -- a public sample distributed
with LandXML.org documentation). All Line/Curve/Alignment values
below are byte-identical to the original file's <Alignments> block;
only the outer <LandXML> root/namespace wrapper was reconstructed to
isolate <Alignments> from the rest of the file (see
GSG_alignments_only_extract.xml) -- the full original file also has
a separate, real finding unrelated to this test: its first
<CgPoints> has no 'name' attribute, which topocore.io.landxml's
PR18B validator currently requires unconditionally. That is a PR18B
question, not addressed here, and is reported separately.

Not synthetic data: this drove the PR18C tolerance-architecture
decision (see session notes). This file has 0 <Spiral> elements and
47 <Curve crvType="arc"> elements across 7 <Alignment>s. 38/47 (81%)
have Start/Center/End values that, combined with the file's own
declared radius, exceed TopoCore's domain tolerance (~1e-9) but are
within the LandXML import tolerance (1e-8) -- confirmed by direct
measurement before this test was written, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topocore.io.landxml.reader import LandXMLReader

_FIXTURE = Path(__file__).parent / "fixtures" / "GSG_alignments_only_extract.xml"


@pytest.mark.skipif(not _FIXTURE.exists(), reason="Real Civil 3D fixture not present")
def test_real_file_reads_without_raising() -> None:
    document = LandXMLReader(_FIXTURE).read()
    assert len(document.alignments) > 0


@pytest.mark.skipif(not _FIXTURE.exists(), reason="Real Civil 3D fixture not present")
def test_real_file_reads_all_seven_alignments() -> None:
    document = LandXMLReader(_FIXTURE).read()
    assert len(document.alignments) == 7


@pytest.mark.skipif(not _FIXTURE.exists(), reason="Real Civil 3D fixture not present")
def test_real_file_conway_farms_drive_has_expected_element_count() -> None:
    document = LandXMLReader(_FIXTURE).read()
    conway = next(a for a in document.alignments if a.name == "Conway Farms Drive")
    # 5 Line + 4 Curve, per the raw XML audited in the session.
    assert len(conway.alignment.elements) == 9


@pytest.mark.skipif(not _FIXTURE.exists(), reason="Real Civil 3D fixture not present")
def test_real_file_reports_tolerance_acceptances_transparently() -> None:
    document, report = LandXMLReader(_FIXTURE).read_with_report()

    tolerance_warnings = [w for w in report.warnings if "import tolerance" in w]

    # Measured independently at audit time: most real curves in this
    # file need the widened tolerance. Assert a substantial fraction
    # do, without hardcoding the exact count (a lower bound is the
    # meaningful assertion -- confirms transparency actually fires
    # on real data, not that a specific brittle count holds forever).
    assert len(tolerance_warnings) >= 30

    for warning in tolerance_warnings:
        assert "Conway Farms Drive" in warning or any(a.name in warning for a in document.alignments)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="Real Civil 3D fixture not present")
def test_real_file_no_element_silently_dropped_for_tolerance_reasons() -> None:
    """
    Every element that needed the widened tolerance is present in
    the resulting Alignment (not silently excluded) -- only crvType/
    spiType/compound-spiral/missing-Center cases are skipped, and
    this file has none of those (confirmed: 0 <Spiral>, all <Curve
    crvType="arc"> with <Center>).
    """
    document, report = LandXMLReader(_FIXTURE).read_with_report()

    skip_warnings = [w for w in report.warnings if "skipped" in w]
    assert len(skip_warnings) == 0

    total_elements = sum(len(a.alignment.elements) for a in document.alignments)
    assert total_elements == 70  # 23 Line + 47 Curve, counted directly from the raw XML


# ----------------------------------------------------------------------
# PLATEIA 2007 real file (Sample_Plateia2007LandXML11.XML).
#
# Unlike the Civil 3D fixture above, this file does NOT read
# end-to-end yet: 1 of its 10 element-to-element station/coordinate
# junctions has a ~1e-6 discrepancy (PREHODNICA 4 -> PREHODNICA 5,
# a probable trailing-zero export anomaly -- 9/10 junctions match
# exactly). Hernán explicitly decided NOT to extend Alignment's
# continuity check with a LandXML import tolerance over a single,
# non-systemic case (see PR18C session notes). This test documents
# that as the current, INTENTIONAL behavior -- not a silent gap --
# so if Alignment is ever extended (or this turns out to need a fix
# some other way), this test's outcome flips from "expects an error"
# to "reads successfully", which is an unambiguous signal that the
# decision changed.
# ----------------------------------------------------------------------

_PLATEIA_FIXTURE = Path(__file__).parent / "fixtures" / "Sample_Plateia2007LandXML11.XML"


@pytest.mark.skipif(not _PLATEIA_FIXTURE.exists(), reason="Real PLATEIA fixture not present")
def test_real_plateia_file_currently_fails_on_known_discontinuity() -> None:
    from topocore.io.landxml.exceptions import LandXMLParseError

    with pytest.raises(LandXMLParseError, match="PREHODNICA 4|element 6 ends at"):
        LandXMLReader(_PLATEIA_FIXTURE).read()


@pytest.mark.skipif(not _PLATEIA_FIXTURE.exists(), reason="Real PLATEIA fixture not present")
def test_real_plateia_file_error_is_cleanly_wrapped_not_a_traceback() -> None:
    """
    Even though this file doesn't fully read yet, the failure must
    still be a clean LandXMLParseError with the offending element
    identified and the original AlignmentGeometryError chained --
    never a raw, uncaught domain exception.
    """
    from topocore.io.landxml.exceptions import LandXMLParseError

    with pytest.raises(LandXMLParseError) as excinfo:
        LandXMLReader(_PLATEIA_FIXTURE).read()

    assert "OS_0" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
