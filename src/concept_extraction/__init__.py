"""Standalone, offline port of the deployed extraction block (steps 1-4).

`run_block(sentences)` reproduces the debug route's
`/api/keywords/debug/benchmark_sentences` with no network involved. The glossary index and
ONNX classifier are loaded from consumer-supplied paths (see `config.py`); the embedder is
fetched on demand from Hugging Face unless configured otherwise.

Importing this package never requires torch, FAISS or the ONNX runtime — those are only
imported lazily inside the step functions.
"""

from .config import PRODUCTION, BlockConfig, validate_artifacts
from .run import BlockResult, run_block

__all__ = ["BlockConfig", "PRODUCTION", "BlockResult", "run_block", "validate_artifacts"]