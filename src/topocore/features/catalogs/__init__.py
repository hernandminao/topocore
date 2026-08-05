"""
topocore.features.catalogs
=============================

Feature-code catalogs shipped with TopoCore -- the wiring point that
assembles the 9 individually-migrated, individually-tested catalogs
into ALL_CODES.

No business logic, no additional validation, no statistics live
here -- those belong to catalogs._validation and
catalogs.catalog_audit respectively.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.features.catalogs.cadastre import CADASTRE_CODES
from topocore.features.catalogs.catalog_audit import (
    AuditViolation,
    CatalogAuditReport,
    run_audit,
)
from topocore.features.catalogs.control import CONTROL_CODES
from topocore.features.catalogs.default import DEFAULT_CODES
from topocore.features.catalogs.drainage import DRAINAGE_CODES
from topocore.features.catalogs.structures import STRUCTURE_CODES
from topocore.features.catalogs.terrain import TERRAIN_CODES
from topocore.features.catalogs.transportation import TRANSPORTATION_CODES
from topocore.features.catalogs.utilities import UTILITY_CODES
from topocore.features.catalogs.vegetation import VEGETATION_CODES
from topocore.features.feature_codes import FeatureCodeDefinition

ALL_CODES: tuple[FeatureCodeDefinition, ...] = (
    *DEFAULT_CODES,
    *TERRAIN_CODES,
    *CONTROL_CODES,
    *VEGETATION_CODES,
    *STRUCTURE_CODES,
    *TRANSPORTATION_CODES,
    *DRAINAGE_CODES,
    *CADASTRE_CODES,
    *UTILITY_CODES,
)

__all__ = [
    "ALL_CODES",
    "AuditViolation",
    "CatalogAuditReport",
    "run_audit",
]
