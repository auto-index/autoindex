## CRITICAL: Corpus Structure

The corpus contains **full Wikipedia articles about named entities** (people, places, organizations, events, etc.). Each entry in `documents.jsonl` is a complete Wikipedia article, potentially several thousand words long with multiple sections.

The `doc_id` is a unique identifier for each article.

**Goal**: Given a natural-language query that specifies multiple attributes the target entity must satisfy simultaneously (e.g. "a musician born in France who won a Grammy before 1990"), find the entity whose article satisfies all stated conditions.
