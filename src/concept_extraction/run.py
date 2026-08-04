"""The extraction block, end to end, offline.

Reproduces `POST /api/keywords/debug/benchmark_sentences` — steps 1-4 of
`extractor.extract_keywords_smart`, driven from pre-segmented sentences:

    1  extract_noun_phrases     NLTK chunker, 2-5 words, [a-z] only
    2  glossary match (FAISS)   name guard + definition/context cosine
    3  ONNX classifier          on everything the glossary missed
    4  rank + top_n             glossary gets a flat +2.0 bonus

Like the route, it does **not** call `split_sentences` — sentences arrive segmented, so the
under-8-word filter never applies. Feeding it a whole document instead of sentences would
silently change the instrument.

Two axes vary independently, which is the point of having this offline:

    rung  — the candidate generator (P0.1's ladder: character class, length bounds, dedup)
    cfg   — everything downstream (glossary thresholds, classifier threshold, top_n)

`run_block(sentences)` with no arguments is the deployed configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .candidates.ablation import LADDER, Rung
from .candidates.chunker import verify_vendored
from .candidates.pipeline import extract_noun_phrases
from .classifier import filter_with_classifier
from .config import PRODUCTION, BlockConfig
from .glossary import match_glossary_faiss

A0 = LADDER[0]


@dataclass
class BlockResult:
    """The route's response, plus what the route throws away."""

    total_sentences: int
    execution_time_seconds: float
    stage_1_unique_nps: int
    stage_2_glossary_matches: int
    stage_3_unmatched_nps_evaluated: int
    stage_3_onnx_classified_positive: int
    total_concepts_returned: int
    concepts: list[dict[str, Any]]

    # --- not exposed by the route -------------------------------------
    noun_phrases: dict[str, list[str]] = field(default_factory=dict)
    classifier_scores: list[tuple[str, float]] = field(default_factory=list)
    config_fingerprint: str = ""
    rung: str = ""
    vendored_sha: str = ""

    def to_route_shape(self) -> dict[str, Any]:
        """Exactly the fields `BenchmarkSentencesResponse` carries — for parity checks."""
        return {
            "total_sentences": self.total_sentences,
            "stage_1_unique_nps": self.stage_1_unique_nps,
            "stage_2_glossary_matches": self.stage_2_glossary_matches,
            "stage_3_unmatched_nps_evaluated": self.stage_3_unmatched_nps_evaluated,
            "stage_3_onnx_classified_positive": self.stage_3_onnx_classified_positive,
            "total_concepts_returned": self.total_concepts_returned,
            "concepts": self.concepts,
        }

    def names(self) -> list[str]:
        return [c["name"] for c in self.concepts]


def rank_keywords(entries: list[dict], cfg: BlockConfig) -> list[dict]:
    """`extractor.rank_keywords` — the flat glossary bonus makes the two sources
    non-comparable by score, which is deliberate in production and worth preserving here."""
    for e in entries:
        bonus = cfg.glossary_rank_bonus if e.get("source") == "glossary" else 0.0
        e["relevance_score"] = round(e.get("semantic_score", 0.0) + bonus, 4)
    entries.sort(key=lambda e: e["relevance_score"], reverse=True)
    return entries


def run_block(
    sentences: list[str],
    cfg: BlockConfig = PRODUCTION,
    rung: Rung = A0,
    *,
    verify: bool = True,
    extra_nps: dict[str, list[str]] | None = None,
) -> BlockResult:
    """Run steps 1-4 over pre-segmented sentences.

    `extra_nps` merges additional candidates into step 1's output before glossary matching,
    for evaluating alternative candidate generators (P0.15's benepar routing) through the
    real steps 2-4 rather than a reimplementation of them. Default `None` leaves the path
    byte-identical to the deployed block; the parity fixture covers that case.
    """
    if not sentences:
        raise ValueError("sentences must not be empty")

    sha = verify_vendored() if verify else ""
    t0 = time.perf_counter()

    # -- step 1 ---------------------------------------------------------
    all_nps: dict[str, list[str]] = {}
    for sent in sentences:
        for np_text in extract_noun_phrases(sent, rung):
            all_nps.setdefault(np_text, []).append(sent)
    if extra_nps:
        for np_text, ctxs in extra_nps.items():
            all_nps.setdefault(np_text, []).extend(ctxs)
    stage_1 = len(all_nps)

    # -- step 2 ---------------------------------------------------------
    if cfg.glossary_matching_strategy != "faiss":
        raise NotImplementedError(
            f"only the deployed 'faiss' strategy is ported; got {cfg.glossary_matching_strategy!r}"
        )
    merged, glossary_matched = match_glossary_faiss(all_nps, cfg)
    stage_2 = len(merged)

    # -- step 3 ---------------------------------------------------------
    unmatched_count = sum(1 for n in all_nps if n.lower() not in glossary_matched)
    merged, scored = filter_with_classifier(all_nps, glossary_matched, merged, cfg)
    stage_3_pos = sum(
        1 for e in merged.values() if e.get("source") == "scientific_concept_classifier"
    )

    # -- step 4 ---------------------------------------------------------
    ranked = rank_keywords(list(merged.values()), cfg)
    for e in ranked:
        if not e.get("paper_context"):
            ctxs = all_nps.get(e.get("name", ""), [])[: cfg.max_contexts]
            e["paper_context"] = " \n...\n ".join(ctxs) if ctxs else ""

    returned = ranked[: cfg.top_n] if cfg.top_n and cfg.top_n > 0 else ranked

    return BlockResult(
        total_sentences=len(sentences),
        execution_time_seconds=round(time.perf_counter() - t0, 4),
        stage_1_unique_nps=stage_1,
        stage_2_glossary_matches=stage_2,
        stage_3_unmatched_nps_evaluated=unmatched_count,
        stage_3_onnx_classified_positive=stage_3_pos,
        total_concepts_returned=len(returned),
        concepts=returned,
        noun_phrases=all_nps,
        classifier_scores=scored,
        config_fingerprint=cfg.fingerprint(),
        rung=rung.name,
        vendored_sha=sha,
    )
