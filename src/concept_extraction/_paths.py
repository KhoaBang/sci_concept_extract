"""Vendored-artifact paths inside the package.

The vendored chunker and its recorded hash ship with the package (as package data), so
they resolve relative to this module. This works identically in a source checkout and in
an installed wheel.
"""

from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent

VENDORED_DIR = _PKG / "_vendored"
VENDORED_CHUNKER = VENDORED_DIR / "nounphrase_extractor.py"
VENDORED_CHUNKER_SHA = VENDORED_DIR / "nounphrase_extractor.sha256"