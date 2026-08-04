"""The vendored chunker, subclassed so each restriction can be lifted independently.

The vendored file is loaded by path rather than imported as a package, because it lives under
`data/vendored/` (an artifact directory, not an import root) and because loading it by path
lets the harness verify its hash on every run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from types import ModuleType

from .._paths import VENDORED_CHUNKER, VENDORED_CHUNKER_SHA
from .ablation import Rung


def _load_vendored() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "concept_extraction._vendored_nounphrase_extractor", VENDORED_CHUNKER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored chunker at {VENDORED_CHUNKER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vendored_sha256() -> str:
    return hashlib.sha256(VENDORED_CHUNKER.read_bytes()).hexdigest()


def verify_vendored() -> str:
    """The instrument must be the one recorded, or the run is not the run it claims to be."""
    actual = vendored_sha256()
    expected = VENDORED_CHUNKER_SHA.read_text().strip()
    if actual != expected:
        raise RuntimeError(
            f"vendored chunker hash mismatch\n  expected {expected}\n  actual   {actual}"
        )
    return actual


_vendored = _load_vendored()
_BaseExtractor = _vendored.CandidateExtractorRegExpNLTK

# The production predicate, copied from the vendored `word_check` so the relaxed variant can
# be expressed as its negation rather than by editing the vendored file.
_PROD_WORD_RE = re.compile(r"^([a-z]+-)*[a-z]+(\s[0-9]+){0,1}$")


def prod_word_check(word: str) -> bool:
    return len(word) > 1 and bool(_PROD_WORD_RE.search(word))


class AblatableExtractor(_BaseExtractor):
    """`CandidateExtractorRegExpNLTK` with the token filter and length range parameterised.

    Only two things vary: `word_check` (called as `self.word_check`, so overridable) and
    `length_range` (read from `self`). The grammar and the chunk-walking logic are the
    vendored ones, untouched.
    """

    def __init__(self, rung: Rung):
        super().__init__(length_range=[rung.min_words, rung.max_words])
        self._accept_any_token = rung.accept_any_token

    def word_check(self, word):  # type: ignore[override]
        if self._accept_any_token:
            return True
        return prod_word_check(word)
