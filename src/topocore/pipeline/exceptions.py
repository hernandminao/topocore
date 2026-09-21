"""
topocore.pipeline.exceptions -- PROPUESTA, no auditada todavia con
la disciplina completa de PR22.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations


class SurveyPipelineError(Exception):
    """Base de todas las excepciones del pipeline."""


class MissingReferenceCodeError(SurveyPipelineError):
    """survey_type=ROAD pero el survey no trae el reference_code declarado (ej. "EJE")."""


class PropertyBoundaryError(SurveyPipelineError):
    """
    El limite de un survey_type=PROPERTY no paso alguna puerta de
    validacion concreta (puntos insuficientes, no cerrado,
    autointerseccion real, area <= 0). El pipeline se detiene en vez
    de generar salidas sobre un limite geometricamente invalido.
    """


class InsufficientDataError(SurveyPipelineError):
    """Datos insuficientes para un paso del pipeline (ej. TIN con menos de 3 puntos GROUND)."""


__all__ = [
    "InsufficientDataError",
    "MissingReferenceCodeError",
    "PropertyBoundaryError",
    "SurveyPipelineError",
]
