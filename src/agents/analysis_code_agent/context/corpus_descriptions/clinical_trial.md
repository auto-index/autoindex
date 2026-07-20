## CRITICAL: Corpus Structure

The corpus contains **full clinical trial descriptions**. Each entry in `documents.jsonl` is a complete clinical trial record, potentially including eligibility criteria, conditions, interventions, inclusion/exclusion criteria, summary, and detailed description sections.

The `doc_id` is a unique identifier for each clinical trial.

**Goal**: Given a natural-language query describing a patient scenario or clinical context, find the clinical trial(s) most relevant to that scenario.
