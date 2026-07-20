"""
bm25_client.py - Thin httpx-based HTTP client for the BM25 FastAPI server.

Each method creates a fresh httpx.Client per call to avoid stale connection issues
during long-running agent loops. A retry decorator handles transient failures.
"""

from __future__ import annotations

import functools
import time

import httpx


def _with_retry(max_attempts: int = 3, backoff: float = 2.0):
    """Decorator: on httpx errors, wait with exponential backoff and retry."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (httpx.RequestError, httpx.TimeoutException) as e:
                    last_exc = e
                except httpx.HTTPStatusError as e:
                    if e.response.status_code < 500:
                        raise  # 4xx: client error, retrying won't help
                    last_exc = e
                if attempt < max_attempts - 1:
                    time.sleep(backoff * (2 ** attempt))
            raise last_exc

        return wrapper

    return decorator


class BM25Client:
    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        timeout: float = 600.0,
        max_retries: int = 3,
        batch_size: int = 100_000,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._batch_size = batch_size

    def build_index(self, name: str, chunks: list, persist: bool = False) -> None:
        """Build a BM25 index on the server.

        For small chunk lists (<=_BATCH_SIZE) uses a single POST.
        For larger lists, streams chunks in batches via append/finalize
        to avoid memory-blowing JSON payloads.
        """
        if len(chunks) <= self._batch_size:
            self._build_index_single(name, chunks, persist)
        else:
            self._build_index_batched(name, chunks, persist)

    @_with_retry()
    def _build_index_single(self, name: str, chunks: list, persist: bool) -> None:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            payload = {
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
                "persist": persist,
            }
            r = client.post(f"/index/{name}/build", json=payload)
            r.raise_for_status()

    def _build_index_batched(self, name: str, chunks: list, persist: bool) -> None:
        """Upload chunks in batches, then finalize to build the index."""
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            self._append_chunks(name, batch)
        self._finalize_index(name, persist)

    @_with_retry()
    def _append_chunks(self, name: str, chunks: list) -> None:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            payload = {
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
            }
            r = client.post(f"/index/{name}/append", json=payload)
            r.raise_for_status()

    @_with_retry()
    def _finalize_index(self, name: str, persist: bool) -> None:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            r = client.post(f"/index/{name}/finalize", json={"persist": persist})
            r.raise_for_status()

    @_with_retry()
    def retrieve(self, name: str, query: str, top_k: int = 100) -> list[dict]:
        """POST /index/{name}/retrieve -> list of {doc_id, score, rank}."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            r = client.post(
                f"/index/{name}/retrieve", json={"query": query, "top_k": top_k}
            )
            r.raise_for_status()
            return r.json()["results"]

    @_with_retry()
    def batch_retrieve(
        self, name: str, queries: list, top_k: int = 100
    ) -> list[dict]:
        """POST /index/{name}/batch_retrieve -> list of {query_id, ranked_docs}.
        queries: list of EvalQuery objects."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            payload = {
                "queries": [
                    {"query_id": q.query_id, "query_text": q.query_text}
                    for q in queries
                ],
                "top_k": top_k,
            }
            r = client.post(f"/index/{name}/batch_retrieve", json=payload)
            r.raise_for_status()
            return r.json()["results"]

    @_with_retry()
    def delete_index(self, name: str) -> None:
        """DELETE /index/{name}."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            r = client.delete(f"/index/{name}")
            r.raise_for_status()

    def health(self) -> bool:
        """GET /health -> True if server is up (no retry; used for polling)."""
        try:
            with httpx.Client(base_url=self.base_url, timeout=5.0) as client:
                r = client.get("/health")
                return r.status_code == 200
        except Exception:
            return False
