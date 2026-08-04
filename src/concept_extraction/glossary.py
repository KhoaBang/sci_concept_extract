"""Step 2 — FAISS glossary matching.

Port of `concept_extraction/glossary_matching_faiss.py::GlossaryFAISS` (loader only) and
`concept_extraction/extractor.py::_match_glossary_faiss` (the matching logic).

The index is read from disk; it is never rebuilt here. Rebuilding would re-embed the
glossary and make the instrument depend on the embedder version, which is exactly the
drift the vendoring is meant to prevent. If `glossary_faiss.index` is missing, that is an
error to surface, not to paper over by building one.
"""

from __future__ import annotations

import functools
import json
import logging
import os

import numpy as np
from rapidfuzz import fuzz

from .config import BlockConfig
from .embed import encode_texts

logger = logging.getLogger(__name__)

_EMBED_DIM = 384
_INDEX_FILENAME = "glossary_faiss.index"
_ENTRIES_FILENAME = "glossary_faiss.json"


class GlossaryFAISS:
    def __init__(self) -> None:
        self.index = None
        self.entries: list[dict] = []

    @staticmethod
    def index_exists(data_folder: str) -> bool:
        return os.path.isfile(os.path.join(data_folder, _INDEX_FILENAME)) and os.path.isfile(
            os.path.join(data_folder, _ENTRIES_FILENAME)
        )

    def load(self, data_folder: str) -> None:
        import faiss

        self.index = faiss.read_index(os.path.join(data_folder, _INDEX_FILENAME))
        with open(os.path.join(data_folder, _ENTRIES_FILENAME), encoding="utf-8") as f:
            self.entries = json.load(f)

    def search_batch(self, query_vecs: np.ndarray, top_k: int = 3):
        if self.index is None or self.index.ntotal == 0:
            empty = np.empty((query_vecs.shape[0], 0), dtype=np.float32)
            return empty, empty.astype(np.int64)
        k = min(top_k, self.index.ntotal)
        return self.index.search(query_vecs.astype(np.float32), k)

    def get_entry(self, idx: int) -> dict | None:
        return self.entries[idx] if 0 <= idx < len(self.entries) else None


@functools.lru_cache(maxsize=4)
def load_glossary_faiss(data_folder: str) -> GlossaryFAISS:
    gf = GlossaryFAISS()
    if not GlossaryFAISS.index_exists(data_folder):
        raise FileNotFoundError(
            f"No prebuilt FAISS glossary index in {data_folder!r}. "
            f"Expected {_INDEX_FILENAME} and {_ENTRIES_FILENAME}. "
            "The harness does not build one — see this module's docstring."
        )
    gf.load(data_folder)
    return gf


def match_glossary_faiss(
    all_noun_phrases: dict[str, list[str]],
    cfg: BlockConfig,
) -> tuple[dict, set[str]]:
    """Verbatim port of `extractor._match_glossary_faiss`.

    Returns (merged, glossary_matched) exactly as the service does — `merged` keyed by
    glossary entry id, `glossary_matched` a set of lowercased NP surface forms.
    """
    merged: dict = {}
    glossary_matched: set[str] = set()

    if not os.path.isdir(cfg.glossary_dir):
        return merged, glossary_matched

    gf = load_glossary_faiss(cfg.glossary_dir)
    if gf.index is None or gf.index.ntotal == 0:
        return merged, glossary_matched

    np_texts = list(all_noun_phrases.keys())
    if not np_texts:
        return merged, glossary_matched

    # -- Phase 1: embed all NPs -------------------------------------------
    np_vecs_list = []
    for i in range(0, len(np_texts), cfg.batch_size):
        np_vecs_list.append(encode_texts(np_texts[i : i + cfg.batch_size], cfg.embed_model))
    np_vecs = np.concatenate(np_vecs_list, axis=0).astype(np.float32)

    # -- Phase 2: nearest-neighbour search --------------------------------
    scores, indices = gf.search_batch(np_vecs, top_k=cfg.faiss_top_k)

    # -- Phase 3: name + length guards ------------------------------------
    candidates: list[tuple[int, dict, list[str], float]] = []
    for ni in range(len(np_texts)):
        np_low = np_texts[ni].lower()
        np_word_count = len(np_low.split())

        for ki in range(scores.shape[1]):
            fscore = float(scores[ni, ki])
            if fscore < cfg.faiss_score_cutoff:
                continue
            entry = gf.get_entry(int(indices[ni, ki]))
            if entry is None or not entry.get("definition"):
                continue

            candidate_names = [entry["name"].lower()]
            for v in entry.get("name_variants", []):
                v_low = v.strip().lower()
                if v_low and v_low != candidate_names[0]:
                    candidate_names.append(v_low)

            best_name_sim = -1.0
            for cand in candidate_names:
                if abs(np_word_count - len(cand.split())) > cfg.max_word_count_diff:
                    continue
                sim = fuzz.ratio(np_low, cand)
                if sim > best_name_sim:
                    best_name_sim = sim

            if best_name_sim < cfg.glossary_fuzzy_threshold:
                continue

            ctxs = all_noun_phrases[np_texts[ni]][: cfg.max_contexts]
            if not ctxs:
                continue
            candidates.append((ni, entry, ctxs, fscore))

    if not candidates:
        return merged, glossary_matched

    # -- Phase 3b: definition <-> context validation ----------------------
    texts_to_encode: list[str] = []
    candidate_meta: list[tuple[int, int, int, dict, float]] = []
    for ni, entry, ctxs, fscore in candidates:
        start_idx = len(texts_to_encode)
        texts_to_encode.append(entry["definition"])
        texts_to_encode.extend(ctxs)
        candidate_meta.append((start_idx, len(ctxs), ni, entry, fscore))

    unique_texts: list[str] = []
    text_to_idx: dict[str, int] = {}
    for t in texts_to_encode:
        if t not in text_to_idx:
            text_to_idx[t] = len(unique_texts)
            unique_texts.append(t)

    unique_vecs_list = []
    for i in range(0, len(unique_texts), cfg.batch_size):
        unique_vecs_list.append(encode_texts(unique_texts[i : i + cfg.batch_size], cfg.embed_model))

    if unique_vecs_list:
        unique_embeddings = np.concatenate(unique_vecs_list, axis=0)
        embeddings = [unique_embeddings[text_to_idx[t]] for t in texts_to_encode]
    else:
        embeddings = []

    best_per_np: dict[int, tuple[float, dict, float]] = {}
    for start_idx, num_ctxs, ni, entry, fscore in candidate_meta:
        def_vec = embeddings[start_idx]
        best_ctx_score = -1.0
        for i in range(num_ctxs):
            score = float(def_vec @ embeddings[start_idx + 1 + i])
            if score > best_ctx_score:
                best_ctx_score = score

        if best_ctx_score < cfg.glossary_def_threshold:
            continue
        if ni not in best_per_np or best_ctx_score > best_per_np[ni][0]:
            best_per_np[ni] = (best_ctx_score, entry, fscore)

    # -- Phase 4: dedup by entry id ---------------------------------------
    for ni, (best_ctx_score, entry, fscore) in best_per_np.items():
        np_text = np_texts[ni]
        eid = entry["id"]
        if eid in merged and merged[eid]["semantic_score"] >= best_ctx_score:
            continue
        merged[eid] = {
            "id": eid,
            "name": np_text,
            "level": entry.get("level", 2),
            "definition": entry.get("definition"),
            "semantic_score": round(best_ctx_score, 4),
            "faiss_score": round(fscore, 4),
            "source": "glossary",
            "definition_source": "glossary",
        }
        glossary_matched.add(np_text.lower())

    return merged, glossary_matched
