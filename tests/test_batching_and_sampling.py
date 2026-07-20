"""
Tests for BM25 batched ingestion (dedup fix) and reservoir sampling in _load_data().

No real corpus or running server required:
- BM25 server tests use FastAPI's in-process TestClient.
- _load_data tests use monkeypatching + the shared tmp_data_dir fixture.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parents[1]
_EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"
for p in [str(_PROJECT_ROOT), str(_EVAL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# BM25 server — append dedup and batched build
# ---------------------------------------------------------------------------

@pytest.fixture
def bm25_test_client():
    """In-process FastAPI TestClient. Resets server state between tests."""
    from fastapi.testclient import TestClient
    import src.agents.analysis_code_agent.bm25_server as srv

    # Clear all in-memory state before each test
    srv._indexes.clear()
    srv._staging.clear()
    srv._staging_ids.clear()
    srv._persist_dir = None

    with TestClient(srv.app) as client:
        yield client

    # Clean up after test too
    srv._indexes.clear()
    srv._staging.clear()
    srv._staging_ids.clear()


def _make_chunk(i: int) -> dict:
    return {"chunk_id": f"chunk_{i}", "doc_id": f"doc_{i}", "text": f"text for chunk {i}", "metadata": {}}


class TestAppendDedup:
    """Sending the same batch twice must not double-stage chunks."""

    def test_duplicate_batch_ignored(self, bm25_test_client):
        client = bm25_test_client
        batch = [_make_chunk(i) for i in range(3)]

        r1 = client.post("/index/test/append", json={"chunks": batch})
        r2 = client.post("/index/test/append", json={"chunks": batch})  # exact retry

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["n_staged"] == 3
        # All 3 were duplicates — nothing new staged
        assert r2.json()["n_staged"] == 3
        assert r2.json()["n_skipped"] == 3

    def test_partial_overlap_adds_only_new(self, bm25_test_client):
        client = bm25_test_client
        first = [_make_chunk(i) for i in range(3)]
        second = [_make_chunk(i) for i in range(2, 5)]  # chunks 2 & 3 are duplicates

        client.post("/index/test/append", json={"chunks": first})
        r = client.post("/index/test/append", json={"chunks": second})

        assert r.json()["n_staged"] == 5        # 3 original + 2 new (chunks 3 & 4)
        assert r.json()["n_skipped"] == 1       # chunk_2 already staged

    def test_finalize_after_retry_builds_correct_index(self, bm25_test_client):
        """End-to-end: duplicate append then finalize → index has deduplicated chunk count."""
        client = bm25_test_client
        batch = [_make_chunk(i) for i in range(4)]

        client.post("/index/test/append", json={"chunks": batch})
        client.post("/index/test/append", json={"chunks": batch})  # retry
        r = client.post("/index/test/finalize", json={"persist": False})

        assert r.status_code == 200
        assert r.json()["n_chunks"] == 4        # not 8

    def test_staging_ids_cleared_after_finalize(self, bm25_test_client):
        """After finalize, starting a new append with the same chunk_ids is allowed."""
        import src.agents.analysis_code_agent.bm25_server as srv
        client = bm25_test_client
        batch = [_make_chunk(i) for i in range(2)]

        client.post("/index/test/append", json={"chunks": batch})
        client.post("/index/test/finalize", json={"persist": False})

        # staging_ids for "test" should be gone
        assert "test" not in srv._staging_ids

        # A fresh append to a new index name with the same chunk_ids should work normally
        r = client.post("/index/test2/append", json={"chunks": batch})
        assert r.json()["n_staged"] == 2
        assert r.json()["n_skipped"] == 0


class TestBatchedBuild:
    """Full batched path (append × N → finalize) produces the same index as a single /build."""

    def test_batched_equals_single_build(self, bm25_test_client):
        client = bm25_test_client
        all_chunks = [_make_chunk(i) for i in range(7)]

        # Single build
        client.post("/index/single/build", json={"chunks": all_chunks, "persist": False})

        # Batched build (3 chunks per batch → 3 append calls)
        batch_size = 3
        for i in range(0, len(all_chunks), batch_size):
            client.post("/index/batched/append", json={"chunks": all_chunks[i:i + batch_size]})
        client.post("/index/batched/finalize", json={"persist": False})

        # Both indexes should return the same result for any query
        q = {"query": "text for chunk 3", "top_k": 5}
        r_single = client.post("/index/single/retrieve", json=q).json()["results"]
        r_batched = client.post("/index/batched/retrieve", json=q).json()["results"]

        assert [x["doc_id"] for x in r_single] == [x["doc_id"] for x in r_batched]

    def test_batched_chunk_count_matches(self, bm25_test_client):
        client = bm25_test_client
        all_chunks = [_make_chunk(i) for i in range(10)]

        for i in range(0, 10, 4):
            client.post("/index/t/append", json={"chunks": all_chunks[i:i + 4]})
        r = client.post("/index/t/finalize", json={"persist": False})

        assert r.json()["n_chunks"] == 10


# ---------------------------------------------------------------------------
# Reservoir sampling in _load_data()
# ---------------------------------------------------------------------------

# Gold doc IDs from the shared fixture (queries.jsonl references these)
_GOLD_IDS = {"doc_001", "doc_002", "doc_003", "doc_005", "doc_007"}
_ALL_IDS = {"doc_001", "doc_002", "doc_003", "doc_004", "doc_005", "doc_006", "doc_007", "doc_008"}
_NON_GOLD_IDS = _ALL_IDS - _GOLD_IDS   # doc_004, doc_006, doc_008


@pytest.fixture
def patched_load_data(tmp_data_dir, monkeypatch):
    """
    Monkeypatch _PROJECT_ROOT in agent.py so _load_data() resolves to tmp_data_dir.

    tmp_data_dir writes files at:
      tmp_data_dir / test_split / documents.jsonl
      tmp_data_dir / test_split / queries.jsonl

    _load_data uses: _PROJECT_ROOT / "data" / split, then falls back to _PROJECT_ROOT / "data".
    We write our own "data/test_split" layout under tmp_path so the primary path hits.
    """
    import src.agents.analysis_code_agent.agent as agent_mod

    # tmp_data_dir is the tmp_path; fixtures are at tmp_path/test_split/
    # _load_data looks for _PROJECT_ROOT/data/<split>  →  we need a "data" subdirectory
    data_root = tmp_data_dir / "data"
    data_root.mkdir(exist_ok=True)
    split_src = tmp_data_dir / "test_split"
    split_dst = data_root / "test_split"
    split_dst.mkdir(exist_ok=True)
    (split_dst / "documents.jsonl").write_bytes((split_src / "documents.jsonl").read_bytes())
    (split_dst / "queries.jsonl").write_bytes((split_src / "queries.jsonl").read_bytes())

    monkeypatch.setattr(agent_mod, "_PROJECT_ROOT", tmp_data_dir)

    from src.agents.analysis_code_agent.agent import _load_data
    return _load_data


class TestReservoirSampling:

    def test_none_loads_all_docs(self, patched_load_data):
        docs, val_queries, eval_queries = patched_load_data("test_split", corpus_size=None)
        assert len(docs) == 8
        assert {d.doc_id for d in docs} == _ALL_IDS

    def test_corpus_size_larger_than_corpus_loads_all(self, patched_load_data):
        docs, val_queries, eval_queries = patched_load_data("test_split", corpus_size=100)
        assert len(docs) == 8

    def test_gold_docs_always_included(self, patched_load_data):
        docs, val_queries, eval_queries = patched_load_data("test_split", corpus_size=6, seed=0)
        doc_ids = {d.doc_id for d in docs}
        assert _GOLD_IDS.issubset(doc_ids), f"Missing gold docs: {_GOLD_IDS - doc_ids}"

    def test_total_count_respects_corpus_size(self, patched_load_data):
        docs, _, _ = patched_load_data("test_split", corpus_size=6, seed=0)
        assert len(docs) == 6

    def test_corpus_size_equals_gold_count_no_non_gold(self, patched_load_data):
        # corpus_size = 5 = exact number of gold docs → no non-gold docs sampled
        docs, _, _ = patched_load_data("test_split", corpus_size=5, seed=0)
        doc_ids = {d.doc_id for d in docs}
        assert doc_ids == _GOLD_IDS

    def test_different_seeds_give_different_non_gold(self, patched_load_data):
        # corpus_size=6 → 5 gold + 1 non-gold. With 3 non-gold to pick from,
        # two different seeds should (occasionally) pick different ones.
        # Run several seed pairs and assert at least one produces a different non-gold set.
        non_gold_sets = set()
        for seed in range(10):
            docs, _, _ = patched_load_data("test_split", corpus_size=6, seed=seed)
            non_gold = frozenset(d.doc_id for d in docs) - _GOLD_IDS
            non_gold_sets.add(non_gold)
        assert len(non_gold_sets) > 1, "All seeds produced the same non-gold sample"

    def test_queries_always_fully_loaded(self, patched_load_data):
        # With only queries.jsonl present, _load_data falls back to using it for
        # both the validation and evaluation query sets.
        _, val_queries, eval_queries = patched_load_data("test_split", corpus_size=5)
        assert len(val_queries) == 5
        assert len(eval_queries) == 5
