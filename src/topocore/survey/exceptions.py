"""
topocore.survey.exceptions
=============================

Exceptions raised by the survey field-data subsystem.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations


class SurveyError(Exception):
    """
    Base exception for the survey subsystem.
    """


class SurveyFormatError(SurveyError):
    """
    Raised when a file's column layout cannot be determined.
    """


class SurveyRecordError(SurveyError):
    """
    Raised when a single record cannot be parsed.
    """


__all__ = [
    "SurveyError",
    "SurveyFormatError",
    "SurveyRecordError",
]
