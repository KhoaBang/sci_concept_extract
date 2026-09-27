# Annotation Harness Research Direction

This folder captures an emerging research direction for scientific concept / terminology extraction:

> **simulate a defensible annotation process, distill that process into a lean annotation harness, then use the resulting supervision to train a small model.**

The immediate scope is deliberately narrow: **automatic term extraction (ATE)** and, in particular, ambiguous judgments around termhood / unithood-like behavior in scientific text.

## Why this exists

The current extraction pipeline ultimately has to decide whether a noun phrase or lexical unit behaves like a scientific concept. A fixed binary classifier can make that decision cheaply, but the target concept itself is hard to define and hard to annotate consistently.

The ACTER work is useful because it treats term annotation as an operational process rather than as a one-line definition. Its annotation scheme separates two axes:

- **lexicon-specificity** — is the lexical unit part of general language or specialist vocabulary?
- **domain-specificity** — is the lexical unit relevant to the domain of the corpus?

Those axes induce three term labels plus a non-term region:

- Specific Term
- Common Term
- Out-of-Domain Term
- Not Term

Named Entities are handled separately.

The ACTER guidelines also contain practical rules for ambiguous cases, recursive/nested terms, split terms, abbreviations, modifiers, units, spelling errors, consistency, use of external evidence, and a "when in doubt, do not annotate" policy.

This gives us something concrete to simulate with LLM agents: **not "ask several LLMs for a label", but simulate an annotation process under a shared operational policy.**

## Core hypothesis

LLMs may be useful as scalable, noisy annotators when:

1. every annotator receives the same explicit operational policy,
2. annotators judge independently before seeing each other's outputs,
3. disagreement is preserved rather than immediately collapsed,
4. hard cases are adjudicated,
5. recurring difficult cases become project memory / guideline refinements,
6. a human-reviewed audit set is kept to measure whether the simulated process tracks human decisions.

The goal is not to treat LLM consensus as ground truth. The goal is to create a reproducible supervision process whose uncertainty and provenance are visible.

## Project shape

```text
Phase 1 — Process simulation
    shared annotation policy
            ↓
    independent annotator agents
            ↓
    disagreement analysis
            ↓
    adjudication + difficult-case ledger
            ↓
    auditable annotation records

Phase 2 — Annotation distillation
    phase-1 process as quality baseline
            ↓
    route easy cases cheaply
            ↓
    escalate uncertain/disputed cases
            ↓
    lean annotation harness

Phase 3 — Model distillation
    lean harness
        ↓
    large supervision set
        ↓
    small scorer / reranker
        ↓
    cheap production inference
```

See [ROADMAP.md](./ROADMAP.md) and [PHASE1_PROTOCOL.md](./PHASE1_PROTOCOL.md).

## Design principles

- **Policy before model.** The annotation contract is a first-class artifact.
- **Independent first-pass judgments.** Do not let annotators anchor on one another before disagreement is measured.
- **Keep ambiguity.** Store disagreement, alternate labels, and difficult-case type instead of flattening everything into one binary target.
- **Separate annotation from downstream policy.** A termhood-like score/ranking can later be thresholded differently for different product needs.
- **Version everything.** Guideline version, prompt, model, decision ledger, and adjudication should be reproducible.
- **Do not genericize too early.** Prove the workflow on ATE before building a generic annotation framework.

## Current status

This is a research seed, not a production framework. The first concrete target is to implement a minimal Phase-1 simulator over ATE candidates and compare its agreement/disagreement behavior against a small human-reviewed set.
