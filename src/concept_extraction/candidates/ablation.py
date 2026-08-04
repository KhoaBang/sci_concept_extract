"""The five restrictions the production candidate generator applies, as independent knobs.

A single before/after pair cannot say which restriction costs what, so P0.1 runs a ladder in
which each rung relaxes exactly one thing. The deltas between adjacent rungs are the prices
of the individual restrictions; the top rung is what the NLTK grammar can produce with every
added restriction removed, and therefore the number that decides whether the generator can
be patched or has to be replaced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# The production value of `split_sentences`' minimum, from
# concept_extraction/text_utils.py:89.
PROD_MIN_SENTENCE_WORDS = 8
# The production value of `extract_noun_phrases`' length_range, from text_utils.py:54.
PROD_MIN_WORDS = 2
PROD_MAX_WORDS = 5
# The ceiling used when max_words is "lifted". Not unbounded: the vendored chunker truncates
# anything longer to its last `max_words` tokens, so an unbounded value would change the
# truncation semantics rather than removing a limit.
LIFTED_MAX_WORDS = 10


@dataclass(frozen=True)
class Rung:
    name: str
    label: str
    # False reproduces `word_check`: [a-z] and internal hyphens only, len > 1.
    # True accepts any token, which lifts the character class AND the single-char rule.
    accept_any_token: bool = False
    min_words: int = PROD_MIN_WORDS
    max_words: int = PROD_MAX_WORDS
    min_sentence_words: int = PROD_MIN_SENTENCE_WORDS
    head_lemma_dedup: bool = True

    def to_json(self) -> dict:
        return asdict(self)


# Each rung relaxes exactly one restriction relative to the one before it.
LADDER: tuple[Rung, ...] = (
    Rung("A0", "production as-is"),
    Rung("A1", "token filter lifted (capitals, digits, single chars)", accept_any_token=True),
    Rung("A2", "+ min_words = 1", accept_any_token=True, min_words=1),
    Rung(
        "A3",
        f"+ max_words = {LIFTED_MAX_WORDS}",
        accept_any_token=True,
        min_words=1,
        max_words=LIFTED_MAX_WORDS,
    ),
    Rung(
        "A4",
        "+ short sentences kept",
        accept_any_token=True,
        min_words=1,
        max_words=LIFTED_MAX_WORDS,
        min_sentence_words=0,
    ),
    Rung(
        "A5",
        "+ head-lemma dedup off  (pure grammar ceiling)",
        accept_any_token=True,
        min_words=1,
        max_words=LIFTED_MAX_WORDS,
        min_sentence_words=0,
        head_lemma_dedup=False,
    ),
)

BY_NAME = {r.name: r for r in LADDER}
