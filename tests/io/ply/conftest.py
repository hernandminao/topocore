from __future__ import annotations

import pytest


@pytest.fixture
def identity_converter():
    """Patch PLYReader converter to return the batch unchanged."""

    def _apply(reader) -> None:
        reader._converter.convert = lambda batch: batch  # type: ignore[method-assign]

    return _apply
