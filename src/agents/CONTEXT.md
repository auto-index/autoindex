# Agent Context

## Dataset

- **Source**: [CRUMB](https://huggingface.co/datasets/jfkback/crumb)
- **Corpus**: Full-length text documents (e.g. complete Wikipedia articles). Each `Document` has:
  - `doc_id` (str): unique identifier
  - `text` (str): full document text (may be several thousand words — chunking is expected)
  - `metadata` (dict): extra fields from the source dataset (may be empty)
- **Queries**: Natural-language queries, each with one or more `relevant_doc_ids`.

The specific split (task) is set at runtime via `--split`. Corpus structure varies by split.

## Preprocessor Interface

Create `src/agents/<name>/preprocess.py` with a class named exactly `Preprocessor`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "my_agent"
    description = "One-line summary"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        ...
```

**Constraints:**
- Return at least one `Chunk` per `Document`
- Each `Chunk.doc_id` must match its source `Document.doc_id`
- `chunk_id` must be globally unique (e.g. `f"{doc_id}_{i}"`)

## Retriever (static – do not modify)

- **Algorithm**: BM25 via `bm25s`
- **Tokeniser**: English stemmer (Snowball)
- Agents control **only** `preprocess()` — the retriever is fixed

Metrics returned: **Recall@k** and **MRR**.
