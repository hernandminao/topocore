"""
topocore.survey.formats
==========================

Named column-order presets for classic total-station / GNSS survey
text formats.

These exist to remove the single most common and most dangerous
mistake in ingesting survey text files: swapping Easting/Northing.
NEZ and ENZ files are positionally identical in every way except
which of the first two columns is X and which is Y -- there is no
way to tell them apart by inspecting the data. Naming the convention
explicitly, instead of asking the caller to figure out
``x_column``/``y_column`` indices by hand, turns a silent,
undetectable blunder into a one-time, explicit choice.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SurveyFormat(Enum):
    """
    Classic headerless survey text column orders.
    """

    #: ID, X, Y, Z, Code
    ID_XYZ_CODE = "id_xyz_code"

    #: Northing, Easting, Z (no point ID)
    NEZ = "nez"

    #: Easting, Northing, Z (no point ID)
    ENZ = "enz"

    #: Punto, Este, Norte, Z
    PENZ = "penz"

    #: Punto, Este, Norte, Z, Descripcion
    PENZD = "penzd"

    #: X, Y, Z, Intensity (no code)
    XYZI = "xyzi"

    #: X, Y, Z, R, G, B (no code)
    XYZRGB = "xyzrgb"


@dataclass(frozen=True, slots=True)
class ColumnLayout:
    """
    Positional column indices for a headerless survey file.

    ``id_column`` and ``code_column`` are ``None`` when the format
    doesn't carry that field.
    """

    x_column: int
    y_column: int
    z_column: int
    id_column: int | None = None
    code_column: int | None = None


_LAYOUTS: dict[SurveyFormat, ColumnLayout] = {
    SurveyFormat.ID_XYZ_CODE: ColumnLayout(
        id_column=0,
        x_column=1,
        y_column=2,
        z_column=3,
        code_column=4,
    ),
    SurveyFormat.NEZ: ColumnLayout(
        x_column=1,
        y_column=0,
        z_column=2,
    ),
    SurveyFormat.ENZ: ColumnLayout(
        x_column=0,
        y_column=1,
        z_column=2,
    ),
    SurveyFormat.PENZ: ColumnLayout(
        id_column=0,
        x_column=1,
        y_column=2,
        z_column=3,
    ),
    SurveyFormat.PENZD: ColumnLayout(
        id_column=0,
        x_column=1,
        y_column=2,
        z_column=3,
        code_column=4,
    ),
    SurveyFormat.XYZI: ColumnLayout(
        x_column=0,
        y_column=1,
        z_column=2,
    ),
    SurveyFormat.XYZRGB: ColumnLayout(
        x_column=0,
        y_column=1,
        z_column=2,
    ),
}


def column_layout(fmt: SurveyFormat) -> ColumnLayout:
    """
    Return the fixed column layout for a named survey format.
    """
    return _LAYOUTS[fmt]


__all__ = [
    "SurveyFormat",
    "ColumnLayout",
    "column_layout",
]
