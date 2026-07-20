import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agents.analysis_code_agent.bm25_client import BM25Client
from src.agents.analysis_code_agent.bm25_server import _indexes, app
from src.agents.analysis_code_agent.code_agent import CodeAgent, Hypothesis
from src.agents.analysis_code_agent.eval_utils import run_subset_eval


class _Resp:
    def __init__(self, response):
        self._response = response
        self.status_code = response.status_code

    def json(self):
        return self._response.json()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _InProcessClient:
    def __init__(self, base_url=None, timeout=None):
        self._tc = TestClient(app)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._tc.close()
        return False

    def post(self, path, json=None):
        return _Resp(self._tc.post(path, json=json))

    def delete(self, path):
        return _Resp(self._tc.delete(path))


class BM25CodeAgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        _indexes.clear()

    def tearDown(self):
        _indexes.clear()

    def test_hypothesis_index_name_and_text_routing_and_eval_names(self):
        baseline_chunk = SimpleNamespace(
            chunk_id="cur_1", doc_id="cur_doc", text="baseline text", metadata={}
        )
        hypothesis_chunk = SimpleNamespace(
            chunk_id="hyp_1", doc_id="hyp_doc", text="hypothesis text", metadata={}
        )
        eval_calls = []

        def fake_eval(index_name, queries, used_client):
            eval_calls.append(index_name)
            if index_name == "hyp_H9":
                self.assertEqual(_indexes["hyp_H9"]["chunks"][0]["text"], "hypothesis text")
                self.assertEqual(_indexes["current"]["chunks"][0]["text"], "baseline text")
            return SimpleNamespace(
                recall_at_100=1.0 if index_name == "hyp_H9" else 0.0,
                recall_at_10=1.0 if index_name == "hyp_H9" else 0.0,
                ndcg_at_10=1.0 if index_name == "hyp_H9" else 0.0,
                per_query=[SimpleNamespace(query_id="q1", hit_at_100=index_name == "hyp_H9")],
            )

        with patch("src.agents.analysis_code_agent.bm25_client.httpx.Client", _InProcessClient):
            client = BM25Client()
            client.build_index("current", [baseline_chunk], persist=False)

            code_agent = CodeAgent({"recall_improvement_threshold": 0.0})
            hypothesis = Hypothesis(id="H9", description="x", rationale="x", code="x")

            with patch.object(code_agent, "_validate_code", return_value=None), patch(
                "src.agents.analysis_code_agent.eval_utils.load_preprocessor_from_code",
                return_value=SimpleNamespace(preprocess=lambda docs: [hypothesis_chunk]),
            ), patch(
                "src.agents.analysis_code_agent.eval_utils.run_subset_eval",
                side_effect=fake_eval,
            ):
                result = code_agent.test_hypothesis(
                    hypothesis=hypothesis,
                    documents=[SimpleNamespace(doc_id="d1", text="t", metadata={})],
                    queries=[SimpleNamespace(query_id="q1", query_text="q1")],
                    current_code="x",
                    client=client,
                )

        self.assertEqual(eval_calls, ["hyp_H9", "current"])
        self.assertNotIn("hyp_H9", _indexes)
        self.assertIn("current", _indexes)
        self.assertIsNone(result.error)

    def test_eval_utils_uses_given_index_name_for_batch_retrieve(self):
        calls = []

        class _Client:
            def batch_retrieve(self, index_name, queries, top_k=100):
                calls.append((index_name, top_k, len(queries)))
                return [{"query_id": "q1", "ranked_docs": [{"doc_id": "d1", "score": 1.0, "rank": 1}]}]

        queries = [SimpleNamespace(query_id="q1", query_text="who", relevant_doc_ids=["d1"])]
        run_subset_eval("hyp_H3", queries, _Client(), top_k=50)

        self.assertEqual(calls, [("hyp_H3", 50, 1)])


if __name__ == "__main__":
    unittest.main()
