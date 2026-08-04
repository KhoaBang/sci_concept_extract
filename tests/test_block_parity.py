"""The offline block must reproduce the deployed service exactly.

If it does not, every scenario run describes a pipeline that is not the one shipped — the
same argument `test_vendor_parity.py` makes for the candidate generator, extended to the
glossary and classifier steps.

The reference is a captured response from the live debug route, stored as a fixture, so
this test needs no network. Recapture it (and say so in the decision log) only when the
service itself changes.

Requires the `inference` extra: `uv sync --extra inference`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "benchmark_sentences_H01-1001.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="no captured service response to check against"
)

COUNTERS = (
    "total_sentences",
    "stage_1_unique_nps",
    "stage_2_glossary_matches",
    "stage_3_unmatched_nps_evaluated",
    "stage_3_onnx_classified_positive",
    "total_concepts_returned",
)


@pytest.fixture(scope="module")
def captured() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def local(captured):
    block = pytest.importorskip(
        "concept_extraction", reason="install the `inference` extra to run parity"
    )
    return block.run_block(captured["request"]["sentences"])


def test_counters_match(local, captured):
    live = captured["response"]
    assert {k: getattr(local, k) for k in COUNTERS} == {k: live[k] for k in COUNTERS}


def test_concept_names_and_order_match(local, captured):
    assert local.names() == [c["name"] for c in captured["response"]["concepts"]]


def test_ids_and_sources_match(local, captured):
    live = {c["name"]: c for c in captured["response"]["concepts"]}
    for c in local.concepts:
        assert c["id"] == live[c["name"]]["id"], c["name"]
        assert c["source"] == live[c["name"]]["source"], c["name"]
        assert c["level"] == live[c["name"]]["level"], c["name"]


@pytest.mark.parametrize("field", ["semantic_score", "faiss_score", "relevance_score"])
def test_scores_match_exactly(local, captured, field):
    """Exact, not approximate. A drifting embedder or a changed threshold shows up here
    as a small delta long before it shows up as a different concept list."""
    live = {c["name"]: c for c in captured["response"]["concepts"]}
    compared = 0
    for c in local.concepts:
        other = live[c["name"]]
        if field in c and field in other:
            assert c[field] == pytest.approx(other[field], abs=1e-9), c["name"]
            compared += 1
    assert compared > 0, f"no concept carried {field}; fixture may be stale"


def test_top_n_is_uncapped_in_the_fixture(captured):
    """The captured request used top_n=-1, so the fixture exercises the whole output
    rather than a truncation. A fixture captured with the default 100 would hide
    ranking bugs on short inputs."""
    assert captured["request"]["top_n"] <= 0
