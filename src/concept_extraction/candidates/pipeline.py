"""The production text → candidate-set path, reimplemented with the ablation knobs exposed.

Faithful to `concept_extraction/text_utils.py` and the loop in `extractor.py:62-71`.
Reimplemented rather than imported because importing the service module pulls in
sentence-transformers, ONNX runtime and the service config. `tests/test_vendor_parity.py`
asserts the A0 rung matches the original implementation's behaviour.

One subtlety worth preserving: the production code calls `extract_noun_phrases` **per
sentence**, so its head-lemma dedup is scoped to a single sentence, and cross-sentence
deduplication happens only through the exact-string key of the accumulating dict. So
"neural networks" and "neural network" occurring in different sentences both survive as
separate candidates, while within one sentence only the first does. Collapsing this to a
document-level dedup would change the candidate count, so it is reproduced as-is.
"""

from __future__ import annotations

from collections import defaultdict

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

from .ablation import Rung
from .chunker import AblatableExtractor

_lemmatizer = WordNetLemmatizer()


def lemmatize_noun_phrase(phrase: str) -> str:
    """Lemmatise the head noun (last word) only — text_utils.py:20."""
    words = phrase.split()
    if not words:
        return phrase
    words[-1] = _lemmatizer.lemmatize(words[-1], pos="n")
    return " ".join(words)


def split_sentences(text: str, min_sentence_words: int = 8) -> list[str]:
    """text_utils.py:80, with the minimum length parameterised."""
    if not text:
        return []
    sentences = []
    for raw_sent in sent_tokenize(text):
        for sent in raw_sent.split("\n\n"):
            sent = sent.strip()
            if len(sent.split()) >= min_sentence_words:
                sentences.append(sent)
    return sentences


def extract_noun_phrases(text: str, rung: Rung) -> list[str]:
    """text_utils.py:54, with the token filter and length range parameterised."""
    extractor = AblatableExtractor(rung)
    raw_nps = extractor(text)

    if not rung.head_lemma_dedup:
        # Still drop exact repeats within the sentence — the vendored extractor already
        # does that via cands_counter, so this only guards the ordering below.
        out, seen = [], set()
        for np_text in sorted(raw_nps, key=len):
            if len(np_text.lower().split()) < rung.min_words:
                continue
            if np_text not in seen:
                seen.add(np_text)
                out.append(np_text)
        return out

    seen_lemmas: set[str] = set()
    unique_nps: list[str] = []
    for np_text in sorted(raw_nps, key=len):
        low = np_text.lower()
        if len(low.split()) < rung.min_words:
            continue
        lemma = lemmatize_noun_phrase(low)
        if lemma not in seen_lemmas and low not in seen_lemmas:
            seen_lemmas.add(lemma)
            seen_lemmas.add(low)
            unique_nps.append(np_text)
    return unique_nps


def generate_candidates(text: str, rung: Rung) -> dict[str, list[str]]:
    """Candidate surface form -> the sentences it was found in. extractor.py:62-71."""
    all_nps: dict[str, list[str]] = defaultdict(list)
    for sent in split_sentences(text, rung.min_sentence_words):
        for np_text in extract_noun_phrases(sent, rung):
            all_nps[np_text].append(sent)
    return dict(all_nps)
