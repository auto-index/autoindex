## CRITICAL: Corpus Structure

The corpus contains **code files or functions**, potentially from open-source repositories. Each entry in `documents.jsonl` is a complete code unit — a function, class, or file — along with any associated docstring or comments.

The `doc_id` is a unique identifier for each code unit.

**Goal**: Given a natural-language query describing a programming task or functionality, find the code unit(s) that implement the described functionality.
