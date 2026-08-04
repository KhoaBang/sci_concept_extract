"""Embedding, reproducing `lol.text.text_model.TextModel.embed_texts` exactly.

The service calls `SentenceTransformer.encode(...)` and then L2-normalises the result with
`sklearn.preprocessing.normalize(axis=1)`. That is numerically equivalent to passing
`normalize_embeddings=True`, but the two-step form is reproduced here so a future change to
either library shows up as a parity failure rather than a silent drift.
"""

from __future__ import annotations

import threading

import numpy as np
from sklearn.preprocessing import normalize as _normalize

from .config import EMBED_MODEL

_model = None
_lock = threading.Lock()


def get_model(model_name: str = EMBED_MODEL):
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(model_name, device="cpu")
    return _model


def encode_texts(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """(N, 384) float32, L2-normalised — the service's `text_utils.encode_texts`."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    vecs = get_model(model_name).encode(texts, show_progress_bar=False)
    vecs = _normalize(vecs, axis=1)
    return np.asarray(vecs, dtype=np.float32)
