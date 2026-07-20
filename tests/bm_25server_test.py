import json
import os
import pathlib
import runpy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.agents.analysis_code_agent import bm25_server as server


class BM25ServerTests(unittest.TestCase):
    def setUp(self):
        server._indexes.clear()
        server._persist_dir = None
        self.client = TestClient(server.app)

    def tearDown(self):
        server._indexes.clear()
        server._persist_dir = None

    def _chunk(self, chunk_id: str, doc_id: str, text: str):
        return {"chunk_id": chunk_id, "doc_id": doc_id, "text": text, "metadata": {}}

    def test_r1_indexes_support_insert_update_delete_and_multiple(self):
        payload_a = {"chunks": [self._chunk("c1", "d1", "alpha")], "persist": False}
        payload_b = {"chunks": [self._chunk("c2", "d2", "beta")], "persist": False}
        payload_a2 = {
            "chunks": [self._chunk("c3", "d3", "gamma"), self._chunk("c4", "d4", "delta")],
            "persist": False,
        }

        r1 = self.client.post("/index/a/build", json=payload_a)
        r2 = self.client.post("/index/b/build", json=payload_b)
        r3 = self.client.post("/index/a/build", json=payload_a2)
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r3.status_code, 200)

        self.assertIn("a", server._indexes)
        self.assertIn("b", server._indexes)
        self.assertSetEqual(set(server._indexes["a"].keys()), {"retriever", "chunks", "n_chunks"})
        self.assertEqual(server._indexes["a"]["n_chunks"], 2)

        delete_ok = self.client.delete("/index/b")
        self.assertEqual(delete_ok.status_code, 200)
        self.assertNotIn("b", server._indexes)

    def test_r1_persist_none_allows_persist_true_without_crash(self):
        mock_retriever = Mock()
        with patch.object(server, "_build_bm25", return_value=mock_retriever):
            r = self.client.post(
                "/index/p/build",
                json={"chunks": [self._chunk("c1", "d1", "alpha")], "persist": True},
            )

        self.assertEqual(r.status_code, 200)
        mock_retriever.save.assert_not_called()

    def test_r1_cli_defaults_and_override_create_dir_and_load_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            persisted_root = tmp_path / "persisted"
            idx_dir = persisted_root / "idx1"
            idx_dir.mkdir(parents=True, exist_ok=True)
            chunk_dicts = [self._chunk("c1", "d1", "Persisted Text")]
            (idx_dir / "chunks.json").write_text(json.dumps(chunk_dicts), encoding="utf-8")

            with patch("argparse.ArgumentParser.parse_args", return_value=SimpleNamespace(port=9999, persist_dir=str(persisted_root))), patch("uvicorn.run") as mock_run:
                mod = runpy.run_module("src.agents.analysis_code_agent.bm25_server", run_name="__main__")

            self.assertTrue(persisted_root.exists())
            self.assertIn("idx1", mod["_indexes"])
            self.assertEqual(mod["_indexes"]["idx1"]["n_chunks"], 1)
            mock_run.assert_called_once()
            self.assertEqual(mock_run.call_args.kwargs["port"], 9999)

        with tempfile.TemporaryDirectory() as tmp2:
            cwd = os.getcwd()
            os.chdir(tmp2)
            try:
                with patch("argparse.ArgumentParser.parse_args", return_value=SimpleNamespace(port=8765, persist_dir=".bm25_cache")), patch("uvicorn.run") as mock_run_default:
                    runpy.run_module("src.agents.analysis_code_agent.bm25_server", run_name="__main__")

                self.assertTrue((pathlib.Path(tmp2) / ".bm25_cache").exists())
                self.assertEqual(mock_run_default.call_args.kwargs["port"], 8765)
            finally:
                os.chdir(cwd)

    def test_r2_build_bm25_lowercases_tokenizes_and_indexes(self):
        mock_retriever = Mock()
        with patch("bm25s.tokenize", return_value=["TOKENS"]) as mock_tokenize, patch("bm25s.BM25", return_value=mock_retriever):
            out = server._build_bm25(["HeLLo", "WoRLD"])

        self.assertIs(out, mock_retriever)
        mock_tokenize.assert_called_once_with(["hello", "world"])
        mock_retriever.index.assert_called_once_with(["TOKENS"])

    def test_r2_empty_chunks_rejected_by_build_endpoint(self):
        r = self.client.post("/index/empty/build", json={"chunks": [], "persist": False})
        self.assertEqual(r.status_code, 400)

    def test_r3_search_logic_query_lowercase_candidate_k_maxp_sort_round_rank(self):
        chunks = [{"doc_id": f"d{i}", "text": f"t{i}"} for i in range(1100)]
        chunks[0]["doc_id"] = "docB"
        chunks[1]["doc_id"] = "docA"
        chunks[2]["doc_id"] = "docA"
        chunks[3]["doc_id"] = "docC"

        retriever = Mock()
        retriever.retrieve.return_value = ([[0, 1, 2, 3]], [[0.7, 0.7, 0.9, 0.123456789]])
        with patch("bm25s.tokenize", return_value=["QTOK"]) as mock_tokenize:
            results = server._search_documents(retriever, chunks, "MiXeD QUERY", top_k=3)

        mock_tokenize.assert_called_once_with(["mixed query"])
        self.assertEqual(retriever.retrieve.call_args.kwargs["k"], 1000)
        self.assertEqual(
            results,
            [
                {"doc_id": "docA", "score": 0.9, "rank": 1},
                {"doc_id": "docB", "score": 0.7, "rank": 2},
                {"doc_id": "docC", "score": 0.123457, "rank": 3},
            ],
        )

    def test_r4_build_replaces_existing_and_stores_dict_chunks(self):
        payload_1 = {"chunks": [self._chunk("c1", "d1", "alpha")], "persist": False}
        payload_2 = {
            "chunks": [self._chunk("c2", "d2", "beta"), self._chunk("c3", "d3", "gamma")],
            "persist": False,
        }
        self.client.post("/index/x/build", json=payload_1)
        self.client.post("/index/x/build", json=payload_2)

        self.assertEqual(server._indexes["x"]["n_chunks"], 2)
        self.assertIsInstance(server._indexes["x"]["chunks"][0], dict)
        self.assertEqual(server._indexes["x"]["chunks"][0]["chunk_id"], "c2")

    def test_r4_build_persist_writes_bm25_and_chunks_json_when_dir_set(self):
        mock_retriever = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            server._persist_dir = pathlib.Path(tmp)
            with patch.object(server, "_build_bm25", return_value=mock_retriever):
                r = self.client.post(
                    "/index/persisted/build",
                    json={"chunks": [self._chunk("c1", "d1", "alpha")], "persist": True},
                )

            self.assertEqual(r.status_code, 200)
            save_path = pathlib.Path(mock_retriever.save.call_args.args[0])
            self.assertEqual(save_path.name, "bm25")
            self.assertTrue((pathlib.Path(tmp) / "persisted" / "chunks.json").exists())

    def test_r5_retrieve_and_batch_endpoints(self):
        server._indexes["idx"] = {"retriever": object(), "chunks": [], "n_chunks": 0}
        with patch.object(server, "_search_documents", side_effect=[
            [{"doc_id": "d1", "score": 1.0, "rank": 1}],
            [{"doc_id": "d2", "score": 0.5, "rank": 1}],
            [{"doc_id": "d3", "score": 0.4, "rank": 1}],
        ]):
            single = self.client.post("/index/idx/retrieve", json={"query": "q", "top_k": 5})
            batch = self.client.post(
                "/index/idx/batch_retrieve",
                json={
                    "queries": [
                        {"query_id": "q1", "query_text": "a"},
                        {"query_id": "q2", "query_text": "b"},
                    ],
                    "top_k": 5,
                },
            )

        self.assertEqual(single.status_code, 200)
        self.assertEqual(single.json(), {"results": [{"doc_id": "d1", "score": 1.0, "rank": 1}]})
        self.assertEqual(batch.status_code, 200)
        self.assertEqual(
            batch.json(),
            {
                "results": [
                    {"query_id": "q1", "ranked_docs": [{"doc_id": "d2", "score": 0.5, "rank": 1}]},
                    {"query_id": "q2", "ranked_docs": [{"doc_id": "d3", "score": 0.4, "rank": 1}]},
                ]
            },
        )

    def test_r5_missing_index_returns_404_for_retrieve_and_batch(self):
        r1 = self.client.post("/index/missing/retrieve", json={"query": "q", "top_k": 10})
        r2 = self.client.post(
            "/index/missing/batch_retrieve",
            json={"queries": [{"query_id": "q1", "query_text": "q"}], "top_k": 10},
        )
        self.assertEqual(r1.status_code, 404)
        self.assertEqual(r2.status_code, 404)

    def test_r6_delete_nonexistent_returns_404(self):
        r = self.client.delete("/index/none")
        self.assertEqual(r.status_code, 404)

    def test_r7_health_and_indexes_metadata(self):
        server._indexes["a"] = {"retriever": object(), "chunks": [self._chunk("c1", "d1", "t")], "n_chunks": 1}
        server._indexes["b"] = {"retriever": object(), "chunks": [self._chunk("c2", "d2", "u"), self._chunk("c3", "d2", "v")], "n_chunks": 2}

        health = self.client.get("/health")
        indexes = self.client.get("/indexes")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertSetEqual(set(health.json()["indexes"]), {"a", "b"})
        self.assertEqual(indexes.status_code, 200)
        self.assertSetEqual(set(indexes.json()["indexes"]), {"a", "b"})
        self.assertEqual(indexes.json()["n_chunks"], {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
