import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import httpx

from src.agents.analysis_code_agent.bm25_client import BM25Client, _with_retry


def _request_error() -> httpx.RequestError:
    return httpx.RequestError("network", request=httpx.Request("GET", "http://x"))


def _timeout_error() -> httpx.TimeoutException:
    return httpx.TimeoutException("timeout")


def _client_cm(client_obj: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__enter__.return_value = client_obj
    cm.__exit__.return_value = False
    return cm


class BM25ClientTests(unittest.TestCase):
    def test_r1_initialization_and_configuration_defaults_and_overrides(self):
        signature = inspect.signature(BM25Client.__init__)
        self.assertIn("base_url", signature.parameters)
        self.assertIn("timeout", signature.parameters)
        # self.assertIn("retries", signature.parameters)
        self.assertEqual(signature.parameters["base_url"].default, "http://localhost:8765")
        self.assertEqual(signature.parameters["timeout"].default, 600.0)
        # self.assertEqual(signature.parameters["retries"].default, 3)

        client = BM25Client(base_url="http://localhost:9999", timeout=30)
        self.assertEqual(client.base_url, "http://localhost:9999")
        self.assertEqual(client.timeout, 30)
        # self.assertEqual(client.retries, 5)

    def test_r2_retry_retries_only_on_requesterror_or_timeout_and_raises_last(self):
        state = {"calls": 0}
        errors = [_request_error(), _timeout_error(), _request_error()]

        @_with_retry(max_attempts=3, backoff=2.0)
        def flaky():
            state["calls"] += 1
            raise errors[state["calls"] - 1]

        with patch("src.agents.analysis_code_agent.bm25_client.time.sleep") as sleep_mock:
            with self.assertRaises(httpx.RequestError):
                flaky()

        self.assertEqual(state["calls"], 3)
        sleep_mock.assert_has_calls([call(2.0), call(4.0)])

    def test_r2_retry_does_not_retry_on_non_retryable_exception(self):
        state = {"calls": 0}

        @_with_retry(max_attempts=5, backoff=2.0)
        def non_retryable():
            state["calls"] += 1
            raise ValueError("no-retry")

        with patch("src.agents.analysis_code_agent.bm25_client.time.sleep") as sleep_mock:
            with self.assertRaises(ValueError):
                non_retryable()

        self.assertEqual(state["calls"], 1)
        sleep_mock.assert_not_called()

    def test_r3_build_index_serializes_chunks_and_posts_and_raises_on_non_200(self):
        chunk = SimpleNamespace(chunk_id="c1", doc_id="d1", text="alpha", metadata={})
        ok_response = MagicMock()
        ok_response.raise_for_status.return_value = None

        inner_client = MagicMock()
        inner_client.post.return_value = ok_response

        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            return_value=_client_cm(inner_client),
        ) as client_ctor:
            BM25Client().build_index("idx", [chunk], persist=True)

        client_ctor.assert_called_once_with(base_url="http://localhost:8765", timeout=600.0)
        inner_client.post.assert_called_once_with(
            "/index/idx/build",
            json={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "doc_id": "d1",
                        "text": "alpha",
                        "metadata": {},
                    }
                ],
                "persist": True,
            },
        )

        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = RuntimeError("non-200")
        inner_client_bad = MagicMock()
        inner_client_bad.post.return_value = bad_response
        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            return_value=_client_cm(inner_client_bad),
        ):
            with self.assertRaises(RuntimeError):
                BM25Client().build_index("idx", [chunk], persist=False)

    def test_r4_retrieve_posts_and_returns_results(self):
        payload = [{"doc_id": "d1", "score": 1.0, "rank": 1}]
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": payload}
        inner_client = MagicMock()
        inner_client.post.return_value = response

        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            return_value=_client_cm(inner_client),
        ):
            out = BM25Client().retrieve("idx", "query", top_k=7)

        inner_client.post.assert_called_once_with(
            "/index/idx/retrieve", json={"query": "query", "top_k": 7}
        )
        self.assertEqual(out, payload)

    def test_r5_batch_retrieve_serializes_queries_and_returns_results(self):
        queries = [
            SimpleNamespace(query_id="q1", query_text="one"),
            SimpleNamespace(query_id="q2", query_text="two"),
        ]
        payload = [{"query_id": "q1", "ranked_docs": [{"doc_id": "d1", "score": 1, "rank": 1}]}]
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": payload}
        inner_client = MagicMock()
        inner_client.post.return_value = response

        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            return_value=_client_cm(inner_client),
        ):
            out = BM25Client().batch_retrieve("idx", queries, top_k=9)

        inner_client.post.assert_called_once_with(
            "/index/idx/batch_retrieve",
            json={
                "queries": [
                    {"query_id": "q1", "query_text": "one"},
                    {"query_id": "q2", "query_text": "two"},
                ],
                "top_k": 9,
            },
        )
        self.assertEqual(out, payload)

    def test_r6_delete_calls_delete_and_raises_on_non_200(self):
        ok_response = MagicMock()
        ok_response.raise_for_status.return_value = None
        inner_client = MagicMock()
        inner_client.delete.return_value = ok_response

        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            return_value=_client_cm(inner_client),
        ):
            BM25Client().delete_index("idx")

        inner_client.delete.assert_called_once_with("/index/idx")

        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = RuntimeError("non-200")
        inner_client_bad = MagicMock()
        inner_client_bad.delete.return_value = bad_response
        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            return_value=_client_cm(inner_client_bad),
        ):
            with self.assertRaises(RuntimeError):
                BM25Client().delete_index("idx")

    def test_r7_health_returns_true_on_200_and_false_on_error(self):
        ok_response = MagicMock(status_code=200)
        inner_client = MagicMock()
        inner_client.get.return_value = ok_response
        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            return_value=_client_cm(inner_client),
        ):
            self.assertTrue(BM25Client().health())

        inner_client_err = MagicMock()
        inner_client_err.get.side_effect = _request_error()
        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            return_value=_client_cm(inner_client_err),
        ):
            self.assertFalse(BM25Client().health())

    def test_r8_fresh_httpx_client_per_method_call(self):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": []}

        inner_clients = []

        def client_factory(*args, **kwargs):
            inner = MagicMock()
            inner.post.return_value = response
            inner_clients.append(inner)
            return _client_cm(inner)

        with patch(
            "src.agents.analysis_code_agent.bm25_client.httpx.Client",
            side_effect=client_factory,
        ) as ctor:
            client = BM25Client()
            client.retrieve("idx", "q1")
            client.retrieve("idx", "q2")

        self.assertEqual(ctor.call_count, 2)
        self.assertEqual(len(inner_clients), 2)
        self.assertIsNot(inner_clients[0], inner_clients[1])


if __name__ == "__main__":
    unittest.main()
