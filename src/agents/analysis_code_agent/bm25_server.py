"""
bm25_server.py - FastAPI BM25 index server.

Hosts named BM25 indexes in memory. Supports building, querying, batch retrieval,
and deletion. Matches the eval harness tokenization (lowercase, bm25s default tokenizer).

Usage:
    python bm25_server.py --port 8765 --persist-dir .bm25_cache
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import bm25s
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="BM25 Index Server")

# ---- In-memory store ----

_indexes: dict[str, dict] = {}
# Each entry: {"retriever": bm25s.BM25, "chunks": list[dict], "n_chunks": int}

# Staging area for batched index builds
_staging: dict[str, list[dict]] = {}
_staging_ids: dict[str, set[str]] = {}  # seen chunk_ids per buffer — deduplicates retried appends

_persist_dir: pathlib.Path | None = None


# ---- Request / response models ----

class ChunkIn(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = {}


class BuildRequest(BaseModel):
    chunks: list[ChunkIn]
    persist: bool = False


class AppendChunksRequest(BaseModel):
    chunks: list[ChunkIn]


class FinalizeRequest(BaseModel):
    persist: bool = False


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 100


class QueryIn(BaseModel):
    query_id: str
    query_text: str


class BatchRetrieveRequest(BaseModel):
    queries: list[QueryIn]
    top_k: int = 100


# ---- Helpers ----

def _build_bm25(texts: list[str]) -> bm25s.BM25:
    """Tokenize and build a BM25 index, matching the eval harness."""
    corpus = [t.lower() for t in texts]
    tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    return retriever


def _search_documents(
    retriever: bm25s.BM25,
    chunks: list[dict],
    query: str,
    top_k: int,
) -> list[dict]:
    """Document-level retrieval with MaxP aggregation, matching eval harness."""
    query_text = query.lower()
    query_tokens = bm25s.tokenize([query_text])

    k = min(len(chunks), 1000)  # candidate_k=1000 like eval harness
    if k == 0:
        return []

    results, scores = retriever.retrieve(query_tokens, k=k)

    hit_indices = [int(x) for x in results[0]]
    hit_scores = [float(x) for x in scores[0]]

    # MaxP aggregation: best chunk score per doc
    doc_scores: dict[str, float] = {}
    for idx, score in zip(hit_indices, hit_scores):
        if 0 <= idx < len(chunks):
            doc_id = chunks[idx]["doc_id"]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

    # Sort by score descending, doc_id as tiebreaker
    sorted_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]

    return [
        {"doc_id": doc_id, "score": round(score, 6), "rank": rank + 1}
        for rank, (doc_id, score) in enumerate(sorted_docs)
    ]


# ---- Endpoints ----

@app.get("/health")
def health():
    return {"status": "ok", "indexes": list(_indexes.keys())}


@app.get("/indexes")
def list_indexes():
    return {
        "indexes": list(_indexes.keys()),
        "n_chunks": {name: entry["n_chunks"] for name, entry in _indexes.items()},
    }


@app.post("/index/{name}/build")
def build_index(name: str, req: BuildRequest):
    texts = [c.text for c in req.chunks]
    if not texts:
        raise HTTPException(400, "No chunks provided")

    retriever = _build_bm25(texts)
    chunk_dicts = [c.model_dump() for c in req.chunks]

    _indexes[name] = {
        "retriever": retriever,
        "chunks": chunk_dicts,
        "n_chunks": len(chunk_dicts),
    }

    # Persist if requested
    if req.persist and _persist_dir is not None:
        save_dir = _persist_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(save_dir / "bm25"))
        with (save_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f)

    return {"status": "built", "n_chunks": len(chunk_dicts)}


@app.post("/index/{name}/retrieve")
def retrieve(name: str, req: RetrieveRequest):
    if name not in _indexes:
        raise HTTPException(404, f"Index '{name}' not found")

    entry = _indexes[name]
    results = _search_documents(
        entry["retriever"], entry["chunks"], req.query, req.top_k
    )
    return {"results": results}


@app.post("/index/{name}/batch_retrieve")
def batch_retrieve(name: str, req: BatchRetrieveRequest):
    if name not in _indexes:
        raise HTTPException(404, f"Index '{name}' not found")

    entry = _indexes[name]
    all_results = []
    for q in req.queries:
        ranked = _search_documents(
            entry["retriever"], entry["chunks"], q.query_text, req.top_k
        )
        all_results.append({"query_id": q.query_id, "ranked_docs": ranked})

    return {"results": all_results}


@app.post("/index/{name}/append")
def append_chunks(name: str, req: AppendChunksRequest):
    """Append chunks to a staging buffer for batched index building.

    Idempotent: duplicate chunk_ids (e.g. from a retried request) are silently skipped.
    """
    if name not in _staging:
        _staging[name] = []
        _staging_ids[name] = set()
    new_chunks = [
        c.model_dump() for c in req.chunks
        if c.chunk_id not in _staging_ids[name]
    ]
    _staging_ids[name].update(c["chunk_id"] for c in new_chunks)
    _staging[name].extend(new_chunks)
    return {"status": "appended", "n_staged": len(_staging[name]), "n_skipped": len(req.chunks) - len(new_chunks)}


@app.post("/index/{name}/finalize")
def finalize_index(name: str, req: FinalizeRequest):
    """Build the BM25 index from all staged chunks, then clear the staging buffer."""
    if name not in _staging or not _staging[name]:
        raise HTTPException(400, f"No staged chunks for index '{name}'")

    chunk_dicts = _staging.pop(name)
    _staging_ids.pop(name, None)
    texts = [c["text"] for c in chunk_dicts]

    retriever = _build_bm25(texts)
    _indexes[name] = {
        "retriever": retriever,
        "chunks": chunk_dicts,
        "n_chunks": len(chunk_dicts),
    }

    if req.persist and _persist_dir is not None:
        save_dir = _persist_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(save_dir / "bm25"))
        with (save_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f)

    return {"status": "built", "n_chunks": len(chunk_dicts)}


@app.delete("/index/{name}")
def delete_index(name: str):
    if name not in _indexes:
        raise HTTPException(404, f"Index '{name}' not found")
    del _indexes[name]
    return {"status": "deleted"}


# ---- CLI entry point ----

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="BM25 Index Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--persist-dir", type=str, default=".bm25_cache")
    args = parser.parse_args()

    _persist_dir = pathlib.Path(args.persist_dir)
    _persist_dir.mkdir(parents=True, exist_ok=True)

    # Load any persisted indexes
    if _persist_dir.exists():
        for idx_dir in _persist_dir.iterdir():
            if idx_dir.is_dir() and (idx_dir / "chunks.json").exists():
                try:
                    with (idx_dir / "chunks.json").open(encoding="utf-8") as f:
                        chunk_dicts = json.load(f)
                    texts = [c["text"] for c in chunk_dicts]
                    retriever = _build_bm25(texts)
                    _indexes[idx_dir.name] = {
                        "retriever": retriever,
                        "chunks": chunk_dicts,
                        "n_chunks": len(chunk_dicts),
                    }
                    print(f"[server] Loaded persisted index '{idx_dir.name}' ({len(chunk_dicts)} chunks)")
                except Exception as e:
                    print(f"[server] Failed to load index '{idx_dir.name}': {e}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
