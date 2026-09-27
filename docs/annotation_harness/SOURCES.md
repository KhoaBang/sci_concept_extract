# Sources and provenance

This research direction is currently grounded in a small number of concrete sources.

## 1. ACTER dataset paper

**Ayla Rigouts Terryn, Véronique Hoste, Els Lefever.**  
*In no uncertain terms: a dataset for monolingual and multilingual automatic term extraction from comparable corpora.*  
Language Resources and Evaluation, 2020.

DOI: https://doi.org/10.1007/s10579-019-09453-9

Why it matters here:

- motivates the difficulty and subjectivity of manual term annotation;
- documents an annotation scheme built around multiple term categories;
- reports iterative pilot annotation, disagreement discussion, guideline refinement, and inter-annotator agreement;
- separates span agreement from label agreement;
- describes final review/consistency checking by an experienced annotator.

## 2. ACTER Terminology Annotation Guidelines

**Ayla Rigouts Terryn.**  
*ACTER Terminology Annotation Guidelines.*  
Version used in this research discussion: March 2021.

Related project resources:

- https://github.com/AylaRT/ACTER
- https://termeval.ugent.be
- UGent handle referenced by the dataset paper: http://hdl.handle.net/1854/LU-8503113

Why it matters here:

- provides the operational policy behind the ACTER annotation process;
- defines lexicon-specificity and domain-specificity;
- gives concrete rules for recursive terms, split terms, abbreviations, modifiers, named entities, consistency, units, misspellings, and ambiguous cases;
- explicitly treats the guidelines as evolving and non-exhaustive;
- recommends recording difficult cases and preserving consistency.

## 3. Contrastive-LM / CLM

Repository: https://github.com/Contrastive-LM/CLM

This is **not** a source for the ACTER annotation protocol. It is a related model-design inspiration for later phases.

The relevant idea is to think about decisions as scoring/ranking compatibility between a state/context and candidate decisions rather than forcing every downstream task into a fixed binary classifier head.

## Source discipline

Phase 1 should keep the following distinction explicit:

- **Source-derived policy:** what ACTER actually specifies.
- **Project adaptation:** choices we make for LLM simulation, candidate-mode annotation, disagreement records, routing, and adjudication.
- **Later modeling hypothesis:** using preference/ranking supervision and a small scorer/reranker.

Do not silently rewrite project adaptations as if they were part of ACTER.
