# Agent Context — Annotation Harness Research

Use this file as a compact handoff for future coding/research agents.

## Problem

The existing scientific concept extraction pipeline contains a classifier stage that must decide whether a candidate phrase behaves like a scientific concept.

The target notion is difficult to label cleanly. Termhood-like judgments are often gradual, domain-sensitive, and ambiguous even for humans. A hard binary target can hide that ambiguity.

## Current research direction

Do not start by training another classifier.

First build an annotation process whose decisions and disagreements are inspectable.

Three phases:

1. **Process simulation** — simulate a human-style ATE annotation panel under a shared operational policy.
2. **Annotation distillation** — make that process cheaper while preserving its decisions.
3. **Model distillation** — use the resulting supervision to train a small scorer/reranker for production.

## Phase 1 baseline

Use ACTER as the initial policy source.

Core decision dimensions:

- lexicon-specificity,
- domain-specificity.

Core term labels:

- Specific Term,
- Common Term,
- Out-of-Domain Term,
- Not Term.

Named Entity is separate when needed.

The first implementation should likely operate on **pre-generated candidates + context** rather than full span extraction, because the product pipeline already has candidate generation and the immediate research question is the quality of the term judgment.

## Panel behavior

- 3 independent first-pass annotators.
- Same guideline and task contract.
- Do not expose one annotator's reasoning to another before first-pass submission.
- Preserve disagreement.
- Adjudicate only after independent judgments are stored.
- Maintain a difficult-case ledger.
- Keep a human-reviewed audit subset.

## Important distinction

This project is **not** trying to prove:

```text
LLM consensus = truth
```

It is trying to test:

```text
Can LLM agents reproduce a defensible annotation process
under an explicit operational policy?
```

And later:

```text
Can that process be compressed cheaply?
Can a small model internalize the useful decision behavior?
```

## Modeling direction

Prefer thinking in terms of scoring/ranking over a polarized binary classifier.

Possible later target:

```text
(context, candidate) → termhood-like score
```

or pairwise supervision:

```text
score(context, A) > score(context, B)
```

The threshold used by a downstream feature should remain a product policy where possible.

## Guardrails

- Do not genericize into a universal annotation framework yet.
- Do not conflate ACTER's policy with our LLM simulation adaptations.
- Do not discard disagreement after majority vote.
- Do not require hidden chain-of-thought from annotator agents.
- Version prompts, guideline references, model identifiers, and adjudication records.
- Measure cost and latency, but only optimize them after the reference process exists.

## Next concrete task

Build a minimal candidate-mode Phase-1 prototype:

1. small set of scientific sentences,
2. candidate noun phrases,
3. 3 independent annotator calls,
4. structured judgments,
5. disagreement analysis,
6. one adjudicator pass for disputed cases,
7. persisted difficult-case ledger,
8. a hand-reviewed audit sample.

Start small enough that every output can be inspected manually.
