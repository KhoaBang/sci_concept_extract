# concept-extraction

A standalone, offline port of the deployed scientific concept extraction block (pipeline
stages 1-4): candidate generation → glossary matching → classifier filter → rank + `top_n`.

This is the **code-only**, distributable version of `extraction_eval`'s `block` package. It
reproduces `POST /api/keywords/debug/benchmark_sentences` with no network and no service
config singleton. It ships **no model weights, glossary index or embedder** — you point it at
your own local artifacts.

## What it does

Given pre-segmented sentences from a paper, it returns the extracted concepts:

```
1  extract_noun_phrases     NLTK chunker, 2-5 words, [a-z] only
2  glossary match (FAISS)   name guard + definition/context cosine
3  ONNX classifier          on everything the glossary missed
4  rank + top_n             glossary gets a flat +2.0 bonus
```

`run_block(sentences)` with no arguments is the deployed configuration. Two axes vary
independently: the candidate-generator `rung` and the downstream `cfg` (thresholds, `top_n`).

## Prerequisites

- **Python ≥ 3.12** and [uv](https://docs.astral.sh/uv/).
- **Artifacts available locally** (paths are configured, never downloaded by default):
  - a **FAISS glossary index** — `glossary_faiss.index` + `glossary_faiss.json` in one dir
  - an **ONNX classifier model dir** — `model.onnx` + tokenizer files
  - the **embedder** — either a Hugging Face model id (fetched on first run, needs network)
    or a local `sentence-transformers` directory
- **Network** on first run only if you use the default Hugging Face embedder.

## Quickstart

```bash
git clone <this-repo> && cd concept_extraction
uv sync --extra inference          # installs ML deps (torch, faiss, ONNX runtime)

# Point at your artifacts (env vars, or the convenience service-checkout root):
export CONCEPT_EXTRACTION_GLOSSARY_DIR=/path/to/glossary
export CONCEPT_EXTRACTION_CLASSIFIER_MODEL=/path/to/onnx_model_dir
# export CONCEPT_EXTRACTION_EMBED_MODEL=all-MiniLM-L6-v2   # optional; default is HF id

# Check your setup:
uv run concept-extract self-check

# Run over sentences:
uv run concept-extract run input.json -o out.json        # or -o out.csv
```

`input.json` is `{"sentences": ["...", "..."]}` (or a plain text file, one sentence per line).

## Configuration reference

| Artifact | Env var | CLI flag |
|---|---|---|
| Glossary dir | `CONCEPT_EXTRACTION_GLOSSARY_DIR` | `--glossary-dir` |
| ONNX classifier dir | `CONCEPT_EXTRACTION_CLASSIFIER_MODEL` | `--classifier-model` |
| Embedder (HF id or local dir) | `CONCEPT_EXTRACTION_EMBED_MODEL` | `--embed-model` |

If you have a `paper-processing-services` checkout, you can instead set only
`PAPER_PROCESSING_ML_ROOT` and the glossary + classifier paths are derived automatically
(`data/glossary`, `model_weights/onnx_minilm_scientific_concept_classifier_model`).

The classifier model choice is **not** hard-coded — v1 vs. a finetuned v2 is just a different
path passed to `CONCEPT_EXTRACTION_CLASSIFIER_MODEL` / `--classifier-model`.

Other knobs (all on the CLI): `--top-n` (default -1 = uncapped), `--threshold` (default 0.60),
`--rung` (default `A0` = production).

## Python API

```python
from concept_extraction import run_block, PRODUCTION, BlockConfig
from dataclasses import replace

result = run_block(sentences)                                    # deployed config
result = run_block(sentences, cfg=replace(PRODUCTION, scientific_concept_threshold=0.9))
result.concepts        # list of dicts: id, name, level, definition, scores, source, context
result.names()         # concept surface forms, in rank order
result.config_fingerprint, result.rung, result.vendored_sha     # provenance
```

Importing the package never loads torch, FAISS or the ONNX runtime — they are imported
lazily inside the step functions.

## Running the tests

```bash
uv sync --extra inference
uv run pytest tests/ -q
```

The parity test (`test_block_parity.py`) compares the offline block against a captured live
response and is skipped if the fixture is missing. It needs the embedder, glossary and
classifier to be reachable (set the env vars above).

## Notes

- The glossary index is **loaded, never built** — a missing index is an error, not something
  to paper over by rebuilding.
- Like the source route, the block does **not** call `split_sentences` — feed it pre-segmented
  sentences. Feeding a whole document silently changes the instrument.
- The vendored `nounphrase_extractor.py` is hash-verified at run time (`verify_vendored`); a
  tampered file fails loudly. See `PROVENANCE.md` for the full provenance of every ported
  module.