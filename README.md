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

### From a PDF (opt-in, requires the `fetch` extra)

If you have PDFs locally, convert them to the per-paper sentence JSON the block expects.
This is fully offline — it only reads the PDFs you point it at, it never downloads anything:

```bash
uv sync --extra fetch               # pymupdf + pymupdf4llm

# a single PDF:
uv run concept-extract fetch paper.pdf -o out_dir

# or a whole directory of PDFs (the PDF pool):
uv run concept-extract fetch papers_dir -o out_dir
```

Each PDF becomes `out_dir/{paper_id}.json` = `{"paper_id": ..., "sentences": [...]}`, where
`paper_id` is the PDF's filename stem. Re-runs skip already-sentenized papers. The sentence
splitter is the same byte-faithful `split_sentences` used by the block (8-word minimum).

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

## How configuration is layered

There is **no `.yml` file** — configuration is a typed, frozen dataclass (`BlockConfig` in
`src/concept_extraction/config.py`) plus environment variables. From lowest to highest
precedence:

1. **`config.py` defaults** — the `BlockConfig` field defaults and the `PRODUCTION` singleton.
   These are the *instrument* and are the deployed values, transcribed from the service's
   `config.yml`. Editing them changes the instrument for **every** run.
2. **Environment variables** — override the three artifact paths (see table above). Read at
   instantiation time via `default_factory`.
3. **CLI flags** — override env for a single run (see below).
4. **`dataclasses.replace(...)`** — the Python API for one-off scenarios without touching
   committed defaults.

The rule (from `config.py`): *"Changing any value here changes the instrument. Do not edit in
place to run a scenario — construct a modified copy with `replace` so the run manifest records
both."* The `config_fingerprint` on every result hashes every active parameter, so a changed
config is detectable in the output.

## Common cases: what to change and where

| What you want | File / flag | How |
|---|---|---|
| **Swap the classifier model** (e.g. v1 → finetuned v2) | env or CLI | Point `CONCEPT_EXTRACTION_CLASSIFIER_MODEL` (or `--classifier-model`) at the other ONNX model dir. The model choice is **not** hard-coded — a different path is a different model. |
| **Change the classifier threshold** (is-it-a-concept cutoff) | CLI, or `config.py` | `--threshold 0.8` (overrides `scientific_concept_threshold`, default 0.60). For library use: `replace(PRODUCTION, scientific_concept_threshold=0.8)`. |
| **Change how many concepts are returned** | CLI, or `config.py` | `--top-n 100` (default -1 = retain full ranking, uncapped). Production behaviour quotes top 100. |
| **Swap the glossary** | env or CLI | Point `CONCEPT_EXTRACTION_GLOSSARY_DIR` (or `--glossary-dir`) at a dir containing `glossary_faiss.index` + `glossary_faiss.json`. The index is **loaded, never built**. |
| **Swap the embedder** | env or CLI | `CONCEPT_EXTRACTION_EMBED_MODEL` (or `--embed-model`) = a Hugging Face id (fetched on first run, needs network) _or_ a local `sentence-transformers` dir. |
| **Use a different candidate generator** | CLI | `--rung <name>` (rungs = ablation ladder; default `A0` = production). List them via `-h`. |
| **Make a permanent default change** | `config.py` | Edit the `BlockConfig` field default (or `PRODUCTION`). This affects all runs and bumps every `config_fingerprint` — record it in a run manifest / decision log. |
| **One-off scenario, no code edit** | Python API | `run_block(sentences, cfg=replace(PRODUCTION, top_n=100, scientific_concept_threshold=0.8))` |

### Concrete example: swap threshold to 0.8 via CLI

```bash
export CONCEPT_EXTRACTION_GLOSSARY_DIR=/path/to/glossary
export CONCEPT_EXTRACTION_CLASSIFIER_MODEL=/path/to/model_dir
uv run concept-extract run input.json -o out.json --threshold 0.8
```

### Concrete example: swap classifier model in Python

```python
from concept_extraction import run_block, PRODUCTION
from dataclasses import replace

cfg = replace(PRODUCTION, classifier_model_path="/data/models/v2_finetuned_onnx")
result = run_block(sentences, cfg=cfg)
print(result.config_fingerprint)  # differs from the default => the change is recorded
```

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
uv sync --extra inference --extra fetch
uv run pytest tests/ -q
```

The parity test (`test_block_parity.py`) compares the offline block against a captured live
response and is skipped if the fixture is missing. It needs the embedder, glossary and
classifier to be reachable (set the env vars above). The PDF tests (`test_fulltext.py`) are
skipped if the `fetch` extra isn't installed.

## Notes

- The glossary index is **loaded, never built** — a missing index is an error, not something
  to paper over by rebuilding.
- Like the source route, the block does **not** call `split_sentences` — feed it pre-segmented
  sentences. Feeding a whole document silently changes the instrument.
- The vendored `nounphrase_extractor.py` is hash-verified at run time (`verify_vendored`); a
  tampered file fails loudly. See `PROVENANCE.md` for the full provenance of every ported
  module.