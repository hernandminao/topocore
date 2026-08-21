"""Regression suite for topocore.features._code_utils.base_code -- PR19. No bugs found."""

from __future__ import annotations

import pytest

from topocore.features._code_utils import base_code


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("CERCA1", "CERCA"),
        ("CERCA", "CERCA"),
        ("MURO123", "MURO"),
        ("123", "123"),  # all-digit code returns unchanged, not empty
        ("A1B2", "A1B"),  # only trailing digits stripped, internal digit kept
        ("", ""),
    ],
)
def test_strips_trailing_digits_only(code: str, expected: str) -> None:
    assert base_code(code) == expected
