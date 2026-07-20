from __future__ import annotations
from typing import List, Tuple, Dict
import bm25s
from schema import Chunk


class BM25Index:
    """
    BM25 index over a list of Chunk objects.

    Defaults are set to match typical IR paper baselines:
    - lowercase=True        : consistent text normalisation
    - candidate_k=1000      : large candidate pool before doc-level aggregation
    - agg="max"             : MaxP (max chunk score per doc) — most common in retrieval papers
    """

    def __init__(
        self,
        chunks: List[Chunk],
        *,
        lowercase: bool = True,
        candidate_k: int = 1000,
        agg: str = "max",
    ) -> None:
        assert agg in ("max", "sum", "avg"), f"agg must be 'max', 'sum' or 'avg', got '{agg}'"

        self._lowercase = lowercase
        self._candidate_k = candidate_k
        self._agg = agg

        # Normalize doc ids to strings
        self._chunks: List[Chunk] = chunks
        for c in self._chunks:
            c.doc_id = str(c.doc_id)

        # Deterministic index -> chunk mapping
        self._idx_to_chunk: Dict[int, Chunk] = {
            i: c for i, c in enumerate(self._chunks)
        }

        # Normalize and tokenize corpus
        corpus_texts = [
            c.text.lower() if self._lowercase else c.text
            for c in self._chunks
        ]
        corpus_tokens = bm25s.tokenize(corpus_texts)

        # Build BM25 index
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)

    # ------------------------------------------------------------------
    # Chunk-level search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Return up to top_k (Chunk, score) hits for the query."""
        query_text = query.lower() if self._lowercase else query
        query_tokens = bm25s.tokenize([query_text])

        k = min(top_k, len(self._chunks))
        results, scores = self._retriever.retrieve(query_tokens, k=k)

        hit_indices = [int(x) for x in results[0]]
        hit_scores  = [float(x) for x in scores[0]]

        return [
            (self._idx_to_chunk[idx], score)
            for idx, score in zip(hit_indices, hit_scores)
        ]

    # ------------------------------------------------------------------
    # Document-level search (aggregates chunk scores -> doc scores)
    # ------------------------------------------------------------------

    def search_documents(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Aggregate chunk-level BM25 scores to document level.

        Aggregation modes:
          max  (MaxP)  — best single chunk score per doc  [paper default]
          sum          — sum of all chunk scores per doc
          avg          — mean chunk score per doc

        Returns list of (doc_id, score) sorted by descending score,
        tie-broken deterministically by doc_id.
        """
        # Retrieve bounded candidate set
        k = min(self._candidate_k, len(self._chunks))
        chunk_hits = self.search(query, top_k=k)

        # Aggregate scores per document
        doc_scores:  Dict[str, float] = {}
        doc_counts:  Dict[str, int]   = {}

        for chunk, score in chunk_hits:
            doc_id = str(chunk.doc_id)
            sc     = float(score)

            if self._agg == "max":
                if doc_id not in doc_scores or sc > doc_scores[doc_id]:
                    doc_scores[doc_id] = sc

            elif self._agg == "sum":
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + sc

            elif self._agg == "avg":
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + sc
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        if self._agg == "avg":
            doc_scores = {
                doc_id: total / doc_counts[doc_id]
                for doc_id, total in doc_scores.items()
            }

        # Deterministic sort: score desc, doc_id asc as tie-breaker
        items = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))
        return items[:top_k]