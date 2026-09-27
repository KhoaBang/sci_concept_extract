# Roadmap

## Phase 1 — Simulate the annotation process

### Question

Can we reproduce a defensible human-style ATE annotation process with LLM agents operating under the same explicit policy?

### Inputs

- scientific text or pre-generated candidate spans,
- corpus domain,
- ACTER-style annotation policy,
- guideline version,
- optional difficult-case ledger.

### Minimal panel

```text
candidate + context + domain + guideline
        │
        ├── Annotator A
        ├── Annotator B
        └── Annotator C
                 ↓
          agreement analysis
                 ↓
        ┌────────┴────────┐
        │                 │
     consensus        disagreement
        │                 │
      accept        adjudicator agent
                          │
                  difficult-case ledger
                          │
                    human audit sample
```

Annotators must make their first-pass decisions independently.

### Judgment record

At minimum:

```json
{
  "candidate": "training objective",
  "context": "...",
  "domain": "machine learning",
  "label": "Common Term",
  "lexicon_specificity": "borderline",
  "domain_specificity": "high",
  "uncertain": true,
  "guideline_refs": ["3.2.1", "3.2.3"],
  "needs_review": true
}
```

The exact schema is not frozen yet. Numeric confidence should not be required until we know it is useful.

### Outputs

- versioned annotation policy,
- raw judgments per annotator,
- consensus/disagreement records,
- adjudication records,
- difficult-case ledger,
- disagreement taxonomy,
- small human audit set,
- cost and latency measurements.

### Success criteria

Phase 1 should answer:

- Do independent LLM annotators produce stable decisions under the same policy?
- Where do they disagree?
- Are disagreements concentrated around recognizable guideline boundaries?
- Does LLM-panel disagreement correlate with human disagreement?
- Which parts of the process actually improve consistency?
- Which recurring cases reveal missing or weak policy?

Do **not** optimize for cost yet.

---

## Phase 2 — Distill the annotation process

### Question

Can we preserve most of Phase-1 quality with a cheaper routing strategy?

Possible cascade:

```text
cheap annotator / scorer
        ↓
clearly easy?
   ├── yes → accept
   └── no
        ↓
strong annotator
        ↓
still uncertain/disputed?
   ├── no → accept
   └── yes → adjudicator or human
```

Possible dimensions to optimize:

- number of model calls,
- model size,
- context size,
- retrieval of guideline sections instead of full guideline injection,
- confidence/disagreement thresholds,
- selective adjudication,
- cached candidate/domain evidence.

### Deliverable

A lean **annotation harness** that reproduces Phase-1 decisions closely enough while reducing cost and latency.

Phase 1 remains the reference process used to evaluate Phase 2.

---

## Phase 3 — Distill the decision function into an SLM

### Question

Can a small model internalize the useful decision behavior of the annotation harness?

The harness generates richer supervision than a binary label:

```text
candidate
context
domain
term category / graded judgment
lexicon-specificity
domain-specificity
pairwise preferences
annotator disagreement
adjudicated outcome
```

The preferred model shape is likely closer to a **scorer/reranker** than a polarized binary classifier.

Example:

```text
(context, candidate) → termhood-like score
```

or a pairwise objective:

```text
score(context, A) > score(context, B)
```

This keeps the downstream threshold as a product policy rather than pretending the ontology itself is binary.

### Production loop

```text
annotation harness
      ↓
supervision dataset
      ↓
small scorer / reranker
      ↓
production
      │
      └── uncertain / hard cases ──→ harness
```

The harness therefore remains useful after distillation as a teacher, evaluator, and escalation path.

---

## Research order

1. Reproduce the ACTER policy faithfully enough for a first simulator.
2. Decide whether Phase 1 operates on full-text span annotation or pre-generated NP candidates.
3. Build a tiny hand-inspected benchmark.
4. Run a 3-annotator panel and inspect disagreement qualitatively.
5. Only then choose agreement metrics and routing heuristics.
6. Do not build a generic annotation framework before the ATE workflow is understood.
