## CRITICAL: Corpus Structure

The corpus contains **~232K full Wikipedia articles** (movies, TV shows, and related topics). Each entry in `documents.jsonl` is a complete article with raw wiki markup (infobox templates, section headers, wiki links, references, etc.).

Documents are typically several thousand words long with multiple sections (Plot/Synopsis, Cast, Production, Reception, Awards, etc.).

The `doc_id` is a unique identifier for each article. Document metadata dicts are EMPTY — any useful info (title, etc.) must be extracted from the document text itself.

**Goal**: Given a "tip of the tongue" query — a vague, colloquial, indirect description of a piece of media (e.g. "Girl in red dress gets chloroformed by shop owner") — find the article describing it.
