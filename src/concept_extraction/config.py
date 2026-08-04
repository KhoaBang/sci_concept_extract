"""Frozen configuration for the vendored extraction block.

Values are transcribed from the deployed service config, not from the dataclass defaults
in `common/settings.py` — the two disagree on three fields and the deployed values are the
ones that produced every number this track has recorded:

    field                     dataclass default    config.yml (deployed)
    glossary_def_threshold    0.35                 0.25
    faiss_top_k               10                   20
    faiss_score_cutoff        0.40                 0.30

Source: paper-processing-services/config.yml, `features.paper_scientific_concept_extraction`.
Transcribed 2026-07-30.

`glossary_dir` and `scientific_concept_classifier_model_path` are container paths
(`/app/...`) in config.yml; they are remapped here to the corresponding locations in the
local checkout. Everything else is verbatim.

Changing any value here changes the instrument. Do not edit in place to run a scenario —
construct a modified copy with `dataclasses.replace` so the run manifest records both.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The embedder is the only artifact that can be fetched on demand (from Hugging Face).
# The glossary index and the ONNX classifier must be provided locally by the consumer.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Artifact path resolution.
#
# This package is code-only: it ships no model weights, glossary index or embedder. The
# consumer must have those artifacts locally and point the package at them. Resolution
# order (first match wins):
#
#   1. CONCEPT_EXTRACTION_GLOSSARY_DIR / CONCEPT_EXTRACTION_CLASSIFIER_MODEL /
#      CONCEPT_EXTRACTION_EMBED_MODEL   — explicit per-artifact paths
#   2. PAPER_PROCESSING_ML_ROOT         — a service-checkout root; glossary and classifier
#      are derived from it exactly as the deployed service lays them out
#   3. empty string                     — not configured; running the block raises a clear
#      error (see `validate_artifacts`)
# ---------------------------------------------------------------------------


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _glossary_dir() -> str:
    explicit = _env("CONCEPT_EXTRACTION_GLOSSARY_DIR")
    if explicit:
        return explicit
    root = _env("PAPER_PROCESSING_ML_ROOT")
    if root:
        return str(Path(root) / "data" / "glossary")
    return ""


def _classifier_model_path() -> str:
    explicit = _env("CONCEPT_EXTRACTION_CLASSIFIER_MODEL")
    if explicit:
        return explicit
    root = _env("PAPER_PROCESSING_ML_ROOT")
    if root:
        return str(
            Path(root) / "model_weights" / "onnx_minilm_scientific_concept_classifier_model"
        )
    return ""


def _embed_model() -> str:
    return _env("CONCEPT_EXTRACTION_EMBED_MODEL") or EMBED_MODEL


@dataclass(frozen=True)
class BlockConfig:
    """One immutable parameter set. Hash it into the run manifest."""

    # -- step 1: candidate generation ------------------------------------
    min_words: int = 2
    max_words: int = 5

    # -- step 2: glossary matching ---------------------------------------
    glossary_matching_strategy: str = "faiss"
    glossary_dir: str = field(default_factory=_glossary_dir)
    glossary_fuzzy_threshold: int = 95
    glossary_def_threshold: float = 0.25
    faiss_top_k: int = 20
    faiss_score_cutoff: float = 0.30
    max_word_count_diff: int = 1

    # -- step 3: concept classifier --------------------------------------
    classifier_model_path: str = field(default_factory=_classifier_model_path)
    scientific_concept_threshold: float = 0.60

    # -- step 4: ranking --------------------------------------------------
    glossary_rank_bonus: float = 2.0
    # DEPLOYED VALUE IS 100 (`extractor.py:35`, applied as `ranked[:top_n]` at :100).
    # -1 is kept as the default here so a run retains the *full* ranking and the cap can be
    # applied in analysis — measuring precision@100 and displacement requires seeing what
    # falls below the cut. Any run quoting production behaviour must slice at 100.
    # Earlier runs on ~100-word abstracts are unaffected: the block emits ~14 terms per
    # abstract, so the cap never binds there. On full papers it binds at 2.5-4.7x.
    top_n: int = -1

    # -- provenance -------------------------------------------------------
    algo_tag: str = "faiss_minilm_v20260404"
    embed_model: str = field(default_factory=_embed_model)
    batch_size: int = 64
    classifier_batch_size: int = 32
    max_contexts: int = 3

    def fingerprint(self) -> str:
        """Stable hash of every parameter, for the run manifest."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


PRODUCTION = BlockConfig()
"""The deployed parameter set. Scenarios are `dataclasses.replace(PRODUCTION, ...)`."""


def validate_artifacts(cfg: BlockConfig) -> list[str]:
    """Return a list of problems with the configured artifact paths (empty = all present).

    The glossary index is loaded, never built, so a missing index is a hard error. The
    classifier is an ONNX model directory. The embedder may be a Hugging Face model id
    (fetched on demand) or a local directory.
    """
    problems: list[str] = []

    if not cfg.glossary_dir:
        problems.append(
            "glossary_dir is not configured — set CONCEPT_EXTRACTION_GLOSSARY_DIR or "
            "PAPER_PROCESSING_ML_ROOT"
        )
    else:
        for name in ("glossary_faiss.index", "glossary_faiss.json"):
            if not (Path(cfg.glossary_dir) / name).is_file():
                problems.append(f"missing glossary file: {Path(cfg.glossary_dir) / name}")

    if not cfg.classifier_model_path:
        problems.append(
            "classifier_model_path is not configured — set "
            "CONCEPT_EXTRACTION_CLASSIFIER_MODEL or PAPER_PROCESSING_ML_ROOT"
        )
    else:
        model_dir = Path(cfg.classifier_model_path)
        if not model_dir.is_dir():
            problems.append(f"classifier model dir not found: {model_dir}")
        elif not (model_dir / "model.onnx").is_file():
            problems.append(f"classifier model dir has no model.onnx: {model_dir}")

    return problems
