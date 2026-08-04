"""The vendored chunker ships with the package and is hash-verified at run time."""

from __future__ import annotations

import hashlib

from concept_extraction._paths import VENDORED_CHUNKER, VENDORED_CHUNKER_SHA
from concept_extraction.candidates.chunker import (
    _load_vendored,
    verify_vendored,
    vendored_sha256,
)


def test_vendored_files_exist():
    assert VENDORED_CHUNKER.is_file()
    assert VENDORED_CHUNKER_SHA.is_file()


def test_recorded_hash_matches_file():
    expected = VENDORED_CHUNKER_SHA.read_text().strip()
    actual = hashlib.sha256(VENDORED_CHUNKER.read_bytes()).hexdigest()
    assert actual == expected


def test_verify_vendored_passes_on_pristine_file():
    assert verify_vendored() == vendored_sha256()


def test_vendored_chunker_loads_and_extracts():
    mod = _load_vendored()
    extractor = mod.CandidateExtractorRegExpNLTK(length_range=[2, 5])
    cands = extractor("This paper studies neural networks and attention mechanisms.")
    assert any("neural networks" in c.lower() for c in cands)