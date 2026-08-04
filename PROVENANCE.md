# The vendored extraction block

An offline, faithful copy of the deployed extraction service's steps 1-4 — the unit §5b of
the foundation names as the thing being improved. It exists so scenarios can be run without
the network, without a debug backend, and without the service's config singleton.

## What it copies, and from where

Service checkout:
`paper_reader/paper-processing-services/paper-processing-ml/src/services/concept_extraction/`

| this module | ported from | notes |
|---|---|---|
| `run.py` | `routes/_gitig_concept_debug_routes.py::benchmark_sentences` | the four steps, in order |
| `run.py::rank_keywords` | `extractor.py::rank_keywords` | flat +2.0 glossary bonus preserved |
| `glossary.py::match_glossary_faiss` | `extractor.py::_match_glossary_faiss` | verbatim |
| `glossary.py::GlossaryFAISS` | `glossary_matching_faiss.py` | **loader only** — never rebuilds |
| `classifier.py::filter_with_classifier` | `extractor.py::_filter_with_classifier` | verbatim |
| `classifier.py::classify_batch` | `concept_classifier.py::classify_scientific_concepts_batch` | thresholding split out |
| `embed.py::encode_texts` | `text_utils.py::encode_texts` → `lol.text.text_model.TextModel` | encode, then sklearn L2-normalise |
| step 1 | `candidates/pipeline.py` + `data/vendored/nounphrase_extractor.py` | reused, already hash-verified |

Config values are transcribed from `paper-processing-services/config.yml`, **not** from the
dataclass defaults in `common/settings.py` — the two disagree on `glossary_def_threshold`
(0.25 vs 0.35), `faiss_top_k` (20 vs 10) and `faiss_score_cutoff` (0.30 vs 0.40). The
deployed values are the ones that produced every recorded number. See `config.py`.

Runtime assets, all inside this user's own checkout:

- glossary index — `paper-processing-ml/data/glossary/glossary_faiss.{index,json}`
- ONNX classifier — `paper-processing-ml/model_weights/onnx_minilm_scientific_concept_classifier_model`
- embedder — `sentence-transformers/all-MiniLM-L6-v2`

Override the checkout root with `PAPER_PROCESSING_ML_ROOT`.

## Deliberate fidelity choices

- **Pre-segmented sentences.** Like the route, `run_block` does not call `split_sentences`,
  so the under-8-word sentence filter never applies. Passing a whole document instead of a
  list of sentences silently changes the instrument.
- **The classifier sees only the first context sentence**, not all three. Production does
  this; it is the mechanism behind aspect A4 and must not be "fixed" here.
- **The glossary index is loaded, never built.** Building would re-embed the glossary and
  make results depend on the embedder version. A missing index raises.
- **`classify_batch` returns every probability**, including for rejected terms, which the
  service discards. This is the only intentional addition: it makes threshold sweeps
  possible without re-running inference. The default path reproduces production exactly.

## Parity

`tests/test_block_parity.py` checks the copy against a captured live-service response
(`tests/fixtures/benchmark_sentences_H01-1001.json`, ACL RD-TEC 2.0 abstract H01-1001).
Verified 2026-07-30: all six funnel counters equal, concept names identical **and in the
same order**, every `id`/`source`/`level` equal, and a maximum score delta of **0.000000**
across `semantic_score`, `faiss_score` and `relevance_score`.

Recapture the fixture only when the service itself changes, and record it in the decision
log — a silently refreshed fixture turns a parity test into a tautology.

## Running scenarios

Two axes vary independently:

```python
from dataclasses import replace
from extraction_eval.block import run_block, PRODUCTION
from extraction_eval.candidates.ablation import LADDER

run_block(sentences)                                   # deployed configuration
run_block(sentences, rung=LADDER[2])                   # candidate generator relaxed
run_block(sentences, cfg=replace(PRODUCTION, scientific_concept_threshold=0.9))
```

`BlockResult.config_fingerprint`, `.rung` and `.vendored_sha` identify the instrument and
belong in any run manifest built on top of this.

Install with the `block` extra — it pulls torch, ONNX runtime and FAISS, which the
candidate-generation experiments deliberately do not need:

```bash
uv run --extra block pytest tests/test_block_parity.py
```
