"""Step 3 — the ONNX scientific-concept classifier.

Port of `concept_extraction/concept_classifier.py::classify_scientific_concepts_batch` and
`concept_extraction/extractor.py::_filter_with_classifier`.

Two details of the production code are reproduced deliberately because they are load-bearing
and easy to "fix" by accident:

- the classifier sees **only the first context sentence** for each NP, not all three;
- `_filter_with_classifier` returns only terms already above threshold, so the service
  never sees the probability of a rejected term. `classify_batch` here returns *every*
  probability and applies the threshold separately, which is what makes threshold sweeps
  possible without re-running inference. The default path reproduces the service exactly.
"""

from __future__ import annotations

import functools

from .config import BlockConfig


@functools.lru_cache(maxsize=2)
def _load(model_path: str):
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForSequenceClassification.from_pretrained(model_path)
    return tok, model


def classify_batch(
    terms: list[str],
    sentences: list[str],
    cfg: BlockConfig,
) -> list[float]:
    """Return P(scientific) for every (term, sentence) pair — no thresholding."""
    if not terms:
        return []
    if len(terms) != len(sentences):
        raise ValueError("terms and sentences must be the same length")

    import torch

    tokenizer, model = _load(cfg.classifier_model_path)
    probs: list[float] = []
    bs = cfg.classifier_batch_size

    for start in range(0, len(terms), bs):
        batch_terms = terms[start : start + bs]
        batch_sents = sentences[start : start + bs]
        inputs = tokenizer(
            batch_terms,
            batch_sents,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        p = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs.extend(p[:, 1].tolist())

    return probs


def filter_with_classifier(
    all_noun_phrases: dict[str, list[str]],
    glossary_matched: set[str],
    merged: dict,
    cfg: BlockConfig,
) -> tuple[dict, list[tuple[str, float]]]:
    """Verbatim port of `extractor._filter_with_classifier`.

    Also returns `(term, probability)` for every NP the classifier scored, including the
    rejected ones — the service discards these, and they are the raw material for a
    threshold sweep.
    """
    unmatched = [
        np_text
        for np_text in all_noun_phrases.keys()
        if np_text.lower() not in glossary_matched
    ]
    if not unmatched:
        return merged, []

    terms: list[str] = []
    contexts: list[str] = []
    for np_text in unmatched:
        ctxs = all_noun_phrases[np_text]
        terms.append(np_text)
        contexts.append(ctxs[0] if ctxs else "")  # first context only — as production does

    probs = classify_batch(terms, contexts, cfg)
    scored = list(zip(terms, probs))

    for np_text, prob in scored:
        if prob < cfg.scientific_concept_threshold:
            continue
        ov_id = f"ov_{np_text.lower().replace(' ', '_')}"
        if ov_id not in merged:
            merged[ov_id] = {
                "id": ov_id,
                "name": np_text,
                "level": -1,
                "definition": "",
                "semantic_score": round(prob, 4),
                "source": "scientific_concept_classifier",
                "definition_source": None,
            }

    return merged, scored
