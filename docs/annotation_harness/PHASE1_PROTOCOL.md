# Phase 1 Protocol — ATE Annotation Panel Simulation

## 1. Scope

Phase 1 simulates an ATE annotation process under a shared operational policy.

It is intentionally **not** a generic annotation framework.

The initial task can be run in either of two modes:

1. **span mode** — annotators identify term spans in running text;
2. **candidate mode** — upstream logic provides candidate lexical units / noun phrases and annotators judge them in context.

For the current scientific-concept pipeline, candidate mode is the likely first implementation because it isolates the ambiguous term judgment from span-detection complexity.

## 2. Shared policy

All annotators receive the same policy.

The initial policy is derived from ACTER:

### Core axes

**Lexicon-specificity**

Is the lexical unit ordinary/common vocabulary, or specialist vocabulary?

**Domain-specificity**

Is the lexical unit relevant to the defined corpus domain?

### Core labels

| Lexicon-specific | Domain-specific | Label |
|---|---|---|
| yes | yes | Specific Term |
| no | yes | Common Term |
| yes | no | Out-of-Domain Term |
| no | no | Not Term |

Named Entities remain a separate category when the task requires them.

## 3. Operational rules to preserve

The initial simulator should preserve the following ACTER-style rules rather than reducing the guideline to a short definition:

- annotate descriptive usage, not only established dictionary/ontology terminology;
- keep the corpus domain explicit;
- allow nested/recursive terms where applicable;
- handle split terms explicitly in span mode;
- maintain consistency across repeated occurrences unless meaning changes;
- abbreviations normally inherit the label of the full form;
- modifiers are included only when they carry a domain-relevant terminological meaning;
- general nouns should not automatically become part of a term;
- spelling mistakes can still be annotated when the intended term remains clear;
- external evidence can be used when the policy allows it;
- when genuinely unsure whether something is a term at all, prefer not to annotate;
- recurring difficult cases should be recorded for consistency.

## 4. Independent first pass

Each annotator receives the same:

- candidate/span,
- local context,
- corpus domain,
- guideline version,
- permitted evidence policy.

Annotator A must not see B/C outputs before committing a first-pass judgment.

Same for B and C.

The point is to measure disagreement, not to create early consensus.

## 5. Suggested judgment schema

```json
{
  "sample_id": "paper42:s18:c3",
  "candidate": "mild anemia",
  "domain": "heart failure",
  "label": "Specific Term",
  "lexicon_specificity": "high",
  "domain_specificity": "high",
  "uncertain": false,
  "guideline_refs": ["3.3.6"],
  "evidence_used": [],
  "needs_review": false
}
```

For an ambiguous case:

```json
{
  "sample_id": "paper17:s9:c4",
  "candidate": "training objective",
  "domain": "machine learning",
  "label": "Common Term",
  "lexicon_specificity": "borderline",
  "domain_specificity": "high",
  "uncertain": true,
  "guideline_refs": ["3.2.1", "3.2.3"],
  "evidence_used": [],
  "needs_review": true
}
```

Do not require long free-form chain-of-thought. Short policy-grounded evidence tags are enough.

## 6. Agreement analysis

Agreement should be decomposed.

In span mode:

- span/boundary agreement,
- category agreement.

In candidate mode:

- category agreement,
- axis-level disagreement,
- uncertainty/disagreement type.

Example:

```text
A → Common Term
B → Specific Term
C → Common Term

panel agreement: 2/3
disagreement axis: lexicon-specificity
```

Do not immediately rewrite this as "gold = Common Term" and discard the disagreement.

## 7. Adjudication

The adjudicator receives:

- the original sample,
- all first-pass judgments,
- guideline,
- difficult-case ledger.

It returns:

- provisional final decision,
- disagreement cause,
- guideline rule used,
- whether the guideline was sufficient,
- whether the case should be added to the difficult-case ledger,
- whether human review is recommended.

The adjudicator is a policy resolver, not an oracle.

## 8. Difficult-case ledger

Example record:

```json
{
  "pattern": "generic academic phrase used heavily inside one field",
  "examples": ["training objective", "experimental setting"],
  "decision": "judge domain-specificity separately from lexicon-specificity",
  "status": "provisional",
  "source": "adjudication batch 2026-09-27"
}
```

The ledger is project-specific memory and should be versioned.

## 9. Human audit

A small human-reviewed subset should include:

- unanimous easy positives,
- unanimous easy negatives,
- 2/1 splits,
- 3-way disagreements,
- recurring difficult-case patterns,
- samples changed by adjudication,
- domain-shift cases.

The important research question is not only "does the panel match the human?" but also:

> **does panel disagreement predict human disagreement?**

## 10. What Phase 1 must not assume

- LLM consensus is not ground truth.
- A single numeric confidence is not necessarily meaningful.
- The ACTER labels are not automatically the final ontology for the product.
- More agents are not automatically better.
- A generic annotation harness should not be designed before this ATE workflow is understood.
