"""
topocore.survey.code_normalization -- PROPUESTA, no auditada todavia
con la disciplina completa de PR22.

Resuelve un patron real de campo confirmado con datos reales
(BUZONES.csv): un activo puede estar codificado con un NUMERO
INDIVIDUAL por cada instancia (ej. "BZ1", "BZ2", ..., "BZ31" -- 31
buzones, cada uno con su propio codigo unico), en vez de un codigo
compartido que necesite dividirse (como CERCA/EST en otros
levantamientos). FeatureCodeRegistry hace coincidencia EXACTA de
texto -- nunca reconoceria "BZ1"..."BZ31" como 31 instancias del
mismo tipo sin 31 entradas identicas en el catalogo, lo cual no es
viable.

Este modulo normaliza el codigo a su prefijo base ANTES de que
FeatureBuilder procese el survey -- "BZ1" se convierte en "BZ", que
si puede registrarse una sola vez en el catalogo. El numero real
("1" de "BZ1") no se pierde: cada punto conserva su propio ``id``
original (el numero de punto real del levantamiento), que
FeatureBuilder ya adjunta a cada Feature como
``attributes["survey_point_ids"]`` -- confirmado, es el mismo
mecanismo que ya usa la etiqueta de puntos en DXF.

Author
------
Hernán Mina

License
-------
MIT
"""
from __future__ import annotations

import dataclasses
import re

from topocore.survey.models import SurveyPoint, SurveyPointSet


def normalize_numbered_codes(survey: SurveyPointSet, prefixes: frozenset[str]) -> SurveyPointSet:
    """
    Para cada prefijo en ``prefixes``, reemplaza cualquier codigo que
    coincida EXACTAMENTE con ``<prefijo><numero>`` (ej. "BZ1", "BZ31")
    por el prefijo solo (ej. "BZ") -- confirmado con datos reales,
    esto es lo que permite registrar un unico codigo en el catalogo
    para todas las instancias numeradas de un mismo tipo de activo.

    Un codigo que ya es EXACTAMENTE el prefijo (ej. un punto con
    codigo "BZ" sin numero) se deja tal cual, sin cambios.

    No modifica ``survey`` -- devuelve un ``SurveyPointSet`` nuevo.

    Parameters
    ----------
    survey
        El survey a normalizar.
    prefixes
        Los prefijos base a reconocer (ej. ``frozenset({"BZ"})``).
    """
    patrones = {prefijo: re.compile(rf"^{re.escape(prefijo)}\d+$") for prefijo in prefixes}

    puntos_normalizados: list[SurveyPoint] = []
    for p in survey.points:
        codigo_nuevo = p.code
        if p.code is not None:
            for prefijo, patron in patrones.items():
                if patron.fullmatch(p.code):
                    codigo_nuevo = prefijo
                    break

        if codigo_nuevo != p.code:
            puntos_normalizados.append(dataclasses.replace(p, code=codigo_nuevo))
        else:
            puntos_normalizados.append(p)

    return dataclasses.replace(survey, points=tuple(puntos_normalizados))


__all__ = ["normalize_numbered_codes"]
