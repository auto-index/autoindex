"""
Tests for the analysis_code_agent pipeline: config, hypotheses, journal, adoption,
analysis agent loop, tools, sanitization, and summary request.

Covers the AnalysisAgent behavioral contracts: config, hypotheses, journal, adoption, tools.
"""

import json
import importlib.util
import pathlib
import sys

import pytest
import yaml
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Make agent packages importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parents[1]
_AGENTS_DIR = _PROJECT_ROOT / "src" / "agents"
_EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"
_AGENT_DIR = _AGENTS_DIR / "analysis_code_agent"

# Add project root so package imports like src.agents... resolve.
for p in [str(_PROJECT_ROOT), str(_EVAL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Keep helper for path-based module loading used in a couple of targeted tests.

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load modules via package imports.
from src.agents.analysis_code_agent.analysis_agent import AnalysisAgent, AnalysisResult
from src.agents.analysis_code_agent.code_agent import CodeAgent, Hypothesis, HypothesisResult
from src.agents.analysis_code_agent.run_journal import RunJournal, IterationRecord, HypothesisRecord
from src.agents.analysis_code_agent.analysis_tools.tools import (
    TOOL_SCHEMAS,
    _validate_path,
    bm25_retrieve,
    dispatch_tool,
    grep_search,
    read_file,
)

import src.agents.analysis_code_agent.analysis_tools.tools as tools_module
import src.agents.analysis_code_agent.analysis_agent as _aa_mod


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def make_llm_response(content="", tool_calls=None):
    """Build a fake litellm completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg
    return resp


def make_tool_call(call_id, name, arguments_dict):
    """Build a fake tool_call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments_dict)
    return tc


def _make_agent(config=None):
    """Create an AnalysisAgent with default config, mocking the system prompt file."""
    if config is None:
        config = {}
    return AnalysisAgent(config)


# =========================================================================
# Step 3: Fill existing 7 stubs
# =========================================================================


class TestConfigLoading:
    def test_loads_yaml(self):
        """Load analysis_code_agent/config.yaml, verify expected keys exist."""
        config_path = _AGENT_DIR / "config.yaml"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        expected_keys = [
            "analysis_model", "analysis_temperature", "code_model",
            "code_temperature", "api_base", "server_port",
            "max_hypotheses", "analysis_max_turns",
            "min_tool_turns",
        ]
        for key in expected_keys:
            assert key in config, f"Missing config key: {key}"


class TestRunJournal:
    def test_add_iteration_and_retrieve(self, tmp_path):
        """Create a RunJournal, add an iteration, verify it appears."""
        journal = RunJournal(tmp_path)
        eval_results = {
            "metrics": {"recall_at_100": 0.80, "ndcg_at_10": 0.65, "recall_at_10": 0.60},
            "query_results": [
                {"query_id": "q1", "hit": True, "rank": 3},
                {"query_id": "q2", "hit": False, "rank": None},
                {"query_id": "q3", "hit": True, "rank": 1},
            ],
        }
        journal.record_iteration(iteration=0, eval_results=eval_results)

        assert len(journal.iterations) == 1
        rec = journal.iterations[0]
        assert rec.iteration == 0
        assert rec.recall_at_100 == 0.80
        assert set(rec.hit_query_ids) == {"q1", "q3"}
        assert rec.miss_query_ids == ["q2"]

    def test_persistent_failure_ids(self, tmp_path):
        """Add multiple iterations where some query_ids always fail."""
        journal = RunJournal(tmp_path)

        # Iteration 0: q1 miss, q2 miss, q3 hit
        journal.record_iteration(0, {"metrics": {"recall_at_100": 0, "ndcg_at_10": 0, "recall_at_10": 0},
            "query_results": [
                {"query_id": "q1", "hit": False}, {"query_id": "q2", "hit": False}, {"query_id": "q3", "hit": True},
            ]})
        # Iteration 1: q1 miss, q2 hit, q3 hit
        journal.record_iteration(1, {"metrics": {"recall_at_100": 0, "ndcg_at_10": 0, "recall_at_10": 0},
            "query_results": [
                {"query_id": "q1", "hit": False}, {"query_id": "q2", "hit": True}, {"query_id": "q3", "hit": True},
            ]})
        # Iteration 2: q1 miss, q2 miss, q3 hit
        journal.record_iteration(2, {"metrics": {"recall_at_100": 0, "ndcg_at_10": 0, "recall_at_10": 0},
            "query_results": [
                {"query_id": "q1", "hit": False}, {"query_id": "q2", "hit": False}, {"query_id": "q3", "hit": True},
            ]})

        persistent = journal.persistent_failure_ids(min_iters=2)
        assert "q1" in persistent  # missed in all 3
        assert "q2" in persistent  # missed in 2
        assert "q3" not in persistent  # never missed


# =========================================================================
# Step 4: New test classes for R1–R8
# =========================================================================


class TestConfigDefaults:
    """R1.1, R1.2, R1.5"""

    def test_defaults_applied(self):
        agent = _make_agent(config={})
        assert agent._model == "openai/gpt-4o-mini"
        assert agent._temperature == 0.3
        assert agent._max_turns == 8
        assert agent._min_tool_turns == 3
        assert agent._api_base == "https://thekeymaker.umass.edu/"

    def test_config_override(self):
        agent = _make_agent(config={
            "analysis_model": "openai/gpt-4o",
            "analysis_temperature": 0.7,
            "analysis_max_turns": 5,
            "min_tool_turns": 2,
            "api_base": "http://custom/",
        })
        assert agent._model == "openai/gpt-4o"
        assert agent._temperature == 0.7
        assert agent._max_turns == 5
        assert agent._min_tool_turns == 2
        assert agent._api_base == "http://custom/"

    def test_tracker_none_valid(self):
        agent = _make_agent(config={})
        assert agent._tracker is None

    def test_system_prompt_loaded(self):
        agent = _make_agent(config={})
        assert isinstance(agent._system_prompt, str)
        assert len(agent._system_prompt) > 100  # non-trivial content

    def test_system_prompt_missing_raises(self, monkeypatch):
        monkeypatch.setattr(_aa_mod, "_AGENT_DIR", pathlib.Path("/nonexistent"))
        with pytest.raises(FileNotFoundError):
            AnalysisAgent(config={})

    def test_default_split_injects_tip_of_the_tongue_corpus_description(self):
        # Default split should resolve to the Wikipedia/tip_of_the_tongue description.
        agent = AnalysisAgent(config={})
        assert "Wikipedia" in agent._system_prompt

    def test_clinical_trial_split_injects_different_corpus_description(self):
        # clinical_trial split should inject clinical trial corpus text, not Wikipedia.
        agent_tot = AnalysisAgent(config={})
        agent_ct = AnalysisAgent(config={}, split="clinical_trial")
        assert agent_tot._system_prompt != agent_ct._system_prompt
        assert "clinical trial" in agent_ct._system_prompt.lower()
        assert "Wikipedia" not in agent_ct._system_prompt


# ---------------------------------------------------------------------------
# R2: _build_candidates
# ---------------------------------------------------------------------------


class TestBuildCandidates:
    """R2.1–R2.6"""

    def _agent(self):
        return _make_agent(config={})

    def test_failures_are_regressions_only(self, mock_eval_results_with_queries, mock_baseline_results):
        agent = self._agent()
        result = agent._build_candidates(mock_eval_results_with_queries, mock_baseline_results)
        failure_ids = [r["query_id"] for r in result["failures"]]
        # q1, q5: baseline hit, current miss → regressions
        assert "q1" in failure_ids
        assert "q5" in failure_ids
        # q3: baseline miss, current miss → NOT a regression
        assert "q3" not in failure_ids
        # q8: not in baseline → NOT a regression
        assert "q8" not in failure_ids

    def test_failures_capped_at_5(self):
        agent = self._agent()
        # 8 regressions
        eval_results = {"query_results": [
            {"query_id": f"q{i}", "hit": False, "rank": None, "relevant_doc_ids": [f"d{i}"], "retrieved_doc_ids": []}
            for i in range(8)
        ]}
        baseline = {"query_results": [
            {"query_id": f"q{i}", "hit": True, "rank": 1}
            for i in range(8)
        ]}
        result = agent._build_candidates(eval_results, baseline)
        assert len(result["failures"]) == 5

    def test_hard_negatives_structure(self, mock_eval_results_with_queries, mock_baseline_results):
        agent = self._agent()
        result = agent._build_candidates(mock_eval_results_with_queries, mock_baseline_results)
        for hn in result["hard_negatives"]:
            assert "query_id" in hn
            assert "wrong_docs" in hn
            assert len(hn["wrong_docs"]) <= 3

    def test_hard_negatives_capped_at_5(self):
        agent = self._agent()
        eval_results = {"query_results": [
            {"query_id": f"q{i}", "hit": False, "rank": None,
             "relevant_doc_ids": [f"d{i}"], "retrieved_doc_ids": [f"wrong_{i}"]}
            for i in range(8)
        ]}
        baseline = {"query_results": []}  # no baseline matches → no regressions, all are plain misses
        result = agent._build_candidates(eval_results, baseline)
        assert len(result["hard_negatives"]) <= 5

    def test_successes_sorted_by_worst_rank(self, mock_eval_results_with_queries, mock_baseline_results):
        agent = self._agent()
        result = agent._build_candidates(mock_eval_results_with_queries, mock_baseline_results)
        ranks = [r.get("rank") or 0 for r in result["successes"]]
        assert ranks == sorted(ranks, reverse=True)

    def test_successes_capped_at_8(self):
        agent = self._agent()
        eval_results = {"query_results": [
            {"query_id": f"q{i}", "hit": True, "rank": i + 1,
             "relevant_doc_ids": [f"d{i}"], "retrieved_doc_ids": [f"d{i}"]}
            for i in range(12)
        ]}
        baseline = {"query_results": []}
        result = agent._build_candidates(eval_results, baseline)
        assert len(result["successes"]) == 8

    def test_empty_query_results(self):
        agent = self._agent()
        result = agent._build_candidates({"query_results": []}, {"query_results": []})
        assert result["failures"] == []
        assert result["hard_negatives"] == []
        assert result["successes"] == []

    def test_missing_query_results_key(self):
        agent = self._agent()
        result = agent._build_candidates({}, {})
        assert result == {"failures": [], "hard_negatives": [], "successes": []}

    def test_no_baseline_match_not_failure(self):
        agent = self._agent()
        eval_results = {"query_results": [
            {"query_id": "q99", "hit": False, "rank": None,
             "relevant_doc_ids": ["d99"], "retrieved_doc_ids": ["d1"]},
        ]}
        baseline = {"query_results": []}  # q99 not in baseline
        result = agent._build_candidates(eval_results, baseline)
        assert len(result["failures"]) == 0

    def test_return_structure_always_three_keys(self):
        agent = self._agent()
        result = agent._build_candidates({}, {})
        assert set(result.keys()) == {"failures", "hard_negatives", "successes"}


# ---------------------------------------------------------------------------
# R3: _build_initial_context
# ---------------------------------------------------------------------------


class TestBuildInitialContext:
    """R3.1–R3.6"""

    def _agent(self):
        return _make_agent(config={})

    def _call(self, agent, **kwargs):
        defaults = {
            "eval_results": {
                "metrics": {"recall_at_100": 0.8000, "ndcg_at_10": 0.6500},
            },
            "baseline_results": {"recall_at_k": 0.5778, "ndcg": 0.1160},
            "current_code": "def preprocess(): pass",
            "candidates": {"failures": [], "hard_negatives": [], "successes": []},
            "split": "tip_of_the_tongue",
            "journal_summary": None,
        }
        defaults.update(kwargs)
        return agent._build_initial_context(**defaults)

    def test_journal_included_when_present(self):
        agent = self._agent()
        output = self._call(agent, journal_summary="Previous run found pattern X")
        assert "Previous run found pattern X" in output

    def test_journal_omitted_when_none(self):
        agent = self._agent()
        output = self._call(agent, journal_summary=None)
        # Should not have an empty journal section
        assert "Previous run" not in output

    def test_metrics_formatted_4_decimal(self):
        agent = self._agent()
        output = self._call(agent)
        assert "0.8000" in output
        assert "0.6500" in output

    def test_current_code_in_python_fence(self):
        agent = self._agent()
        output = self._call(agent, current_code="def preprocess(): pass")
        assert "```python" in output
        assert "def preprocess(): pass" in output

    def test_all_candidate_sections_present(self):
        agent = self._agent()
        output = self._call(agent)
        assert "Regressions" in output or "regressions" in output.lower()
        assert "Hard negatives" in output or "hard negatives" in output.lower()
        assert "Successes" in output or "successes" in output.lower()

    def test_tool_descriptions_present(self):
        agent = self._agent()
        output = self._call(agent)
        assert "bm25_retrieve" in output
        assert "read_file" in output
        assert "grep_search" in output
        assert "curl" not in output.lower()

    def test_data_path_uses_split(self):
        agent = self._agent()
        output = self._call(agent, split="my_custom_split")
        assert "data/my_custom_split/" in output


# ---------------------------------------------------------------------------
# R4: analyze loop
# ---------------------------------------------------------------------------


class TestAnalyzeLoop:
    """R4.1–R4.10"""

    def _make_agent_and_deps(self, config_overrides=None):
        config = {"analysis_max_turns": 8, "min_tool_turns": 3}
        if config_overrides:
            config.update(config_overrides)
        agent = _make_agent(config=config)
        mock_client = MagicMock()
        return agent, mock_client

    @patch("src.agents.analysis_code_agent.analysis_agent.dispatch_tool", return_value="tool result")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_min_tool_turns_nudge(self, mock_comp, mock_dispatch):
        """No tool calls before min_tool_turns → nudge appended."""
        agent, client = self._make_agent_and_deps()

        # Turn 0: no tool calls, no summary → should nudge
        # Turn 1: tool call
        # Turn 2: tool call
        # Turn 3: tool call
        # Turn 4: summary
        mock_comp.side_effect = [
            make_llm_response(content="Let me think about this..."),  # no tools, no summary
            make_llm_response(tool_calls=[make_tool_call("tc1", "bm25_retrieve", {"query": "test"})]),
            make_llm_response(tool_calls=[make_tool_call("tc2", "read_file", {"file_path": "documents.jsonl"})]),
            make_llm_response(tool_calls=[make_tool_call("tc3", "grep_search", {"pattern": "x", "file_path": "f"})]),
            make_llm_response(content="<summary>Final analysis</summary>"),
        ]

        result = agent.analyze(
            eval_results={"metrics": {"recall_at_100": 0.8, "ndcg_at_10": 0.6}, "query_results": []},
            baseline_results={"recall_at_k": 0.5, "ndcg": 0.1, "query_results": []},
            current_code="pass", client=client,
        )

        # Check nudge message was added
        nudge_msgs = [m for m in result.conversation if m.get("role") == "user" and "tool-using turns" in m.get("content", "")]
        assert len(nudge_msgs) >= 1
        assert "0/3" in nudge_msgs[0]["content"] or "0" in nudge_msgs[0]["content"]

    @patch("src.agents.analysis_code_agent.analysis_agent.dispatch_tool", return_value="tool result")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_summary_detected_via_tag(self, mock_comp, mock_dispatch):
        agent, client = self._make_agent_and_deps({"min_tool_turns": 1})

        mock_comp.side_effect = [
            make_llm_response(tool_calls=[make_tool_call("tc1", "bm25_retrieve", {"query": "test"})]),
            make_llm_response(content="Here is my analysis. <summary>Key finding: X is broken</summary>"),
        ]

        result = agent.analyze(
            eval_results={"metrics": {"recall_at_100": 0.8, "ndcg_at_10": 0.6}, "query_results": []},
            baseline_results={"recall_at_k": 0.5, "ndcg": 0.1, "query_results": []},
            current_code="pass", client=client,
        )
        assert result.summary == "Key finding: X is broken"

    @patch("src.agents.analysis_code_agent.analysis_agent.dispatch_tool", return_value="tool result")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_multiple_tool_calls_per_turn(self, mock_comp, mock_dispatch):
        agent, client = self._make_agent_and_deps({"min_tool_turns": 1})

        mock_comp.side_effect = [
            make_llm_response(tool_calls=[
                make_tool_call("tc1", "bm25_retrieve", {"query": "q1"}),
                make_tool_call("tc2", "read_file", {"file_path": "documents.jsonl"}),
            ]),
            make_llm_response(content="<summary>Done</summary>"),
        ]

        result = agent.analyze(
            eval_results={"metrics": {"recall_at_100": 0.8, "ndcg_at_10": 0.6}, "query_results": []},
            baseline_results={"recall_at_k": 0.5, "ndcg": 0.1, "query_results": []},
            current_code="pass", client=client,
        )

        tool_msgs = [m for m in result.conversation if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

    @patch("src.agents.analysis_code_agent.analysis_agent.dispatch_tool", return_value="tool result")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_max_turns_cap(self, mock_comp, mock_dispatch):
        agent, client = self._make_agent_and_deps({"analysis_max_turns": 3, "min_tool_turns": 0})

        mock_comp.side_effect = [
            make_llm_response(tool_calls=[make_tool_call(f"tc{i}", "bm25_retrieve", {"query": "q"})]) for i in range(10)
        ] + [make_llm_response(content="<summary>fallback</summary>")]

        result = agent.analyze(
            eval_results={"metrics": {"recall_at_100": 0.8, "ndcg_at_10": 0.6}, "query_results": []},
            baseline_results={"recall_at_k": 0.5, "ndcg": 0.1, "query_results": []},
            current_code="pass", client=client,
        )

        # Count only assistant messages from the main loop (exclude _request_summary additions)
        # With max_turns=3, the loop runs at most 3 iterations producing 3 assistant messages.
        # _request_summary may add 1-2 more. So total assistant msgs <= 5.
        assistant_msgs = [m for m in result.conversation if m.get("role") == "assistant"]
        # The main loop should produce at most max_turns (3) assistant messages
        # _request_summary adds at most 2 more
        assert len(assistant_msgs) <= 5

    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_llm_none_breaks_loop(self, mock_comp):
        agent, client = self._make_agent_and_deps()

        mock_comp.return_value = make_llm_response(content="<summary>fallback summary</summary>")
        # Override _call_llm to return None on second call
        call_count = [0]
        original_call_llm = agent._call_llm

        def fake_call_llm(messages, turn, tools=None):
            call_count[0] += 1
            if call_count[0] >= 2:
                return None
            return original_call_llm(messages, turn, tools=tools)

        agent._call_llm = fake_call_llm

        result = agent.analyze(
            eval_results={"metrics": {"recall_at_100": 0.8, "ndcg_at_10": 0.6}, "query_results": []},
            baseline_results={"recall_at_k": 0.5, "ndcg": 0.1, "query_results": []},
            current_code="pass", client=client,
        )
        # Should still produce a result (via _request_summary or from existing summary)
        assert result is not None

    @patch("src.agents.analysis_code_agent.analysis_agent.dispatch_tool", return_value="tool result")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_conversation_history_complete(self, mock_comp, mock_dispatch):
        agent, client = self._make_agent_and_deps({"min_tool_turns": 1})

        mock_comp.side_effect = [
            make_llm_response(tool_calls=[make_tool_call("tc1", "bm25_retrieve", {"query": "test"})]),
            make_llm_response(content="<summary>Done</summary>"),
        ]

        result = agent.analyze(
            eval_results={"metrics": {"recall_at_100": 0.8, "ndcg_at_10": 0.6}, "query_results": []},
            baseline_results={"recall_at_k": 0.5, "ndcg": 0.1, "query_results": []},
            current_code="pass", client=client,
        )

        roles = {m["role"] for m in result.conversation}
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    @patch("src.agents.analysis_code_agent.analysis_agent.dispatch_tool", return_value="tool result")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_turn_count_is_assistant_messages(self, mock_comp, mock_dispatch):
        agent, client = self._make_agent_and_deps({"min_tool_turns": 1})

        mock_comp.side_effect = [
            make_llm_response(tool_calls=[make_tool_call("tc1", "bm25_retrieve", {"query": "test"})]),
            make_llm_response(content="<summary>Done</summary>"),
        ]

        result = agent.analyze(
            eval_results={"metrics": {"recall_at_100": 0.8, "ndcg_at_10": 0.6}, "query_results": []},
            baseline_results={"recall_at_k": 0.5, "ndcg": 0.1, "query_results": []},
            current_code="pass", client=client,
        )

        expected = len([m for m in result.conversation if m.get("role") == "assistant"])
        assert result.turns == expected

    @patch("src.agents.analysis_code_agent.analysis_agent.dispatch_tool", return_value="tool result")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_request_summary_called_when_no_tag(self, mock_comp, mock_dispatch):
        """If no <summary> found, _request_summary is called."""
        agent, client = self._make_agent_and_deps({"analysis_max_turns": 2, "min_tool_turns": 0})

        # Two turns of tool calls, no summary tag
        mock_comp.side_effect = [
            make_llm_response(tool_calls=[make_tool_call("tc1", "bm25_retrieve", {"query": "q"})]),
            make_llm_response(tool_calls=[make_tool_call("tc2", "bm25_retrieve", {"query": "q"})]),
            # _request_summary will call completion again
            make_llm_response(content="<summary>Requested summary</summary>"),
        ]

        result = agent.analyze(
            eval_results={"metrics": {"recall_at_100": 0.8, "ndcg_at_10": 0.6}, "query_results": []},
            baseline_results={"recall_at_k": 0.5, "ndcg": 0.1, "query_results": []},
            current_code="pass", client=client,
        )

        assert result.summary == "Requested summary"


# ---------------------------------------------------------------------------
# R5: Tools
# ---------------------------------------------------------------------------


class TestBM25RetrieveTool:
    """R5.A"""

    def test_basic_retrieve(self):
        client = MagicMock()
        client.retrieve.return_value = [{"doc_id": "d1", "score": 5.0, "rank": 1}]
        result = bm25_retrieve(client, "test query")
        parsed = json.loads(result)
        assert parsed[0]["doc_id"] == "d1"
        client.retrieve.assert_called_once_with(name="current", query="test query", top_k=10)

    def test_error_handling(self):
        client = MagicMock()
        client.retrieve.side_effect = RuntimeError("connection failed")
        result = bm25_retrieve(client, "test query")
        assert "Error" in result


class TestReadFileTool:
    """R5.B"""

    @pytest.fixture
    def data_dir(self, tmp_path, monkeypatch):
        """Set up a fake data directory and monkeypatch _PROJECT_ROOT."""
        split_dir = tmp_path / "data" / "test_split"
        split_dir.mkdir(parents=True)
        (split_dir / "documents.jsonl").write_text(
            '{"doc_id": "doc_001", "text": "hello world"}\n'
            '{"doc_id": "doc_002", "text": "foo bar baz"}\n',
            encoding="utf-8",
        )
        (split_dir / "small.txt").write_text("short content", encoding="utf-8")
        monkeypatch.setattr(tools_module, "_PROJECT_ROOT", tmp_path)
        return tmp_path

    def test_reads_file(self, data_dir):
        result = read_file(split="test_split", file_path="small.txt")
        assert "short content" in result

    def test_truncation(self, data_dir):
        result = read_file(split="test_split", file_path="documents.jsonl", max_chars=20)
        assert len(result) <= 20 + len("... [truncated]") + 5
        assert "truncated" in result

    def test_jsonl_filter_by_doc_id(self, data_dir):
        result = read_file(split="test_split", file_path="documents.jsonl", filter_id="doc_001")
        assert "doc_001" in result
        assert "doc_002" not in result

    def test_path_traversal_rejected(self, data_dir):
        result = read_file(split="test_split", file_path="../../../etc/passwd")
        assert "Error" in result or "escapes" in result

    def test_missing_file(self, data_dir):
        result = read_file(split="test_split", file_path="nonexistent.jsonl")
        assert "not found" in result or "Error" in result


class TestGrepSearchTool:
    """R5.C"""

    @pytest.fixture
    def data_dir(self, tmp_path, monkeypatch):
        split_dir = tmp_path / "data" / "test_split"
        split_dir.mkdir(parents=True)
        (split_dir / "documents.jsonl").write_text(
            '{"doc_id": "doc_001", "text": "hello world"}\n'
            '{"doc_id": "doc_002", "text": "foo bar baz"}\n'
            '{"doc_id": "doc_003", "text": "hello again"}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(tools_module, "_PROJECT_ROOT", tmp_path)
        return tmp_path

    def test_basic_grep(self, data_dir):
        result = grep_search(split="test_split", pattern="doc_001", file_path="documents.jsonl")
        assert "doc_001" in result
        assert "line 1" in result

    def test_max_results_cap(self, data_dir):
        result = grep_search(split="test_split", pattern="doc_", file_path="documents.jsonl", max_results=2)
        lines = [l for l in result.strip().split("\n") if l.startswith("line")]
        assert len(lines) == 2

    def test_invalid_regex(self, data_dir):
        result = grep_search(split="test_split", pattern="[invalid", file_path="documents.jsonl")
        assert "invalid regex" in result.lower() or "Error" in result

    def test_path_restriction(self, data_dir):
        result = grep_search(split="test_split", pattern="x", file_path="../secret.txt")
        assert "Error" in result or "escapes" in result


class TestToolDispatch:
    """R5.D"""

    def test_dispatch_routes_bm25(self):
        client = MagicMock()
        client.retrieve.return_value = [{"doc_id": "d1", "score": 1.0, "rank": 1}]
        result = dispatch_tool("bm25_retrieve", {"query": "test"}, client=client, split="s")
        client.retrieve.assert_called_once()

    def test_unknown_tool_returns_error(self):
        result = dispatch_tool("rm_rf", {}, client=MagicMock(), split="s")
        assert "unknown tool" in result.lower() or "Error" in result

    def test_tool_schemas_valid_format(self):
        assert len(TOOL_SCHEMAS) == 3
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]
            assert "required" in schema["function"]["parameters"]


# ---------------------------------------------------------------------------
# R6: _call_llm
# ---------------------------------------------------------------------------


class TestCallLLM:
    """R6.1–R6.7"""

    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_tracker_records_call(self, mock_comp):
        tracker = MagicMock()
        agent = AnalysisAgent(config={}, tracker=tracker)
        mock_comp.return_value = make_llm_response(content="hello")

        agent._call_llm([{"role": "user", "content": "test"}], turn=0, tools=TOOL_SCHEMAS)
        tracker.record_llm_call.assert_called_once()

    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_tools_passed_to_completion(self, mock_comp):
        agent = _make_agent(config={})
        mock_comp.return_value = make_llm_response(content="hello")

        agent._call_llm([{"role": "user", "content": "test"}], turn=0, tools=TOOL_SCHEMAS)
        call_kwargs = mock_comp.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == TOOL_SCHEMAS

    @patch("src.agents.analysis_code_agent.analysis_agent.time.sleep")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_content_policy_retry_sanitizes(self, mock_comp, mock_sleep):
        agent = _make_agent(config={})

        class ContentPolicyViolationError(Exception):
            pass

        mock_comp.side_effect = [ContentPolicyViolationError("blocked"), make_llm_response(content="ok")]

        messages = [
            {"role": "user", "content": "test"},
            {"role": "tool", "tool_call_id": "tc1", "content": "x" * 500},
        ]
        result = agent._call_llm(messages, turn=0)
        assert result is not None
        # Messages should have been sanitized in-place
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert len(tool_msg["content"]) <= 420  # 400 + truncation notice

    @patch("src.agents.analysis_code_agent.analysis_agent.time.sleep")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_generic_retry(self, mock_comp, mock_sleep):
        agent = _make_agent(config={})
        mock_comp.side_effect = [RuntimeError("transient"), make_llm_response(content="ok")]

        result = agent._call_llm([{"role": "user", "content": "test"}], turn=0)
        assert result is not None
        mock_sleep.assert_called_once_with(5)

    @patch("src.agents.analysis_code_agent.analysis_agent.time.sleep")
    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_returns_none_on_exhausted_retries(self, mock_comp, mock_sleep):
        agent = _make_agent(config={})
        mock_comp.side_effect = RuntimeError("persistent failure")

        result = agent._call_llm([{"role": "user", "content": "test"}], turn=0)
        assert result is None


# ---------------------------------------------------------------------------
# R7: _sanitize_messages
# ---------------------------------------------------------------------------


class TestSanitizeMessages:
    """R7.1–R7.4"""

    def _agent(self):
        return _make_agent(config={})

    def test_only_tool_results_truncated(self):
        agent = self._agent()
        messages = [
            {"role": "user", "content": "x" * 500},
            {"role": "tool", "tool_call_id": "tc1", "content": "y" * 500},
        ]
        sanitized = agent._sanitize_messages(messages)
        assert len(sanitized[0]["content"]) == 500  # user msg unchanged
        assert len(sanitized[1]["content"]) < 500  # tool msg truncated

    def test_short_tool_result_unchanged(self):
        agent = self._agent()
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": "short"}]
        sanitized = agent._sanitize_messages(messages)
        assert sanitized[0]["content"] == "short"

    def test_threshold_boundary(self):
        agent = self._agent()
        msg_400 = [{"role": "tool", "tool_call_id": "tc1", "content": "a" * 400}]
        msg_401 = [{"role": "tool", "tool_call_id": "tc2", "content": "a" * 401}]
        assert agent._sanitize_messages(msg_400)[0]["content"] == "a" * 400
        assert "truncated" in agent._sanitize_messages(msg_401)[0]["content"]

    def test_returns_new_list(self):
        agent = self._agent()
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": "x" * 500}]
        original_content = messages[0]["content"]
        sanitized = agent._sanitize_messages(messages)
        assert sanitized is not messages
        assert messages[0]["content"] == original_content  # original unchanged


# ---------------------------------------------------------------------------
# R8: _request_summary
# ---------------------------------------------------------------------------


class TestRequestSummary:
    """R8.1–R8.5"""

    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_summary_prompt_appended(self, mock_comp):
        agent = _make_agent(config={})
        mock_comp.return_value = make_llm_response(content="<summary>The summary</summary>")

        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "ctx"}]
        agent._request_summary(messages)

        # A user message requesting summary should have been sent to the LLM
        sent_messages = mock_comp.call_args.kwargs["messages"]
        user_msgs = [m for m in sent_messages if m["role"] == "user"]
        assert any("summary" in m["content"].lower() for m in user_msgs)

    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_tool_call_guardrail(self, mock_comp):
        agent = _make_agent(config={})

        # First response has tool_calls, second is text only
        resp1 = make_llm_response(content="partial", tool_calls=[make_tool_call("tc1", "bm25_retrieve", {"query": "q"})])
        resp2 = make_llm_response(content="<summary>Final answer</summary>")
        mock_comp.side_effect = [resp1, resp2]

        messages = [{"role": "system", "content": "sys"}]
        result = agent._request_summary(messages)

        assert mock_comp.call_count == 2
        # Should have appended a stricter instruction in the second call
        sent_messages = mock_comp.call_args.kwargs["messages"]
        user_msgs = [m for m in sent_messages if m["role"] == "user"]
        assert any("do not attempt tool calls" in m["content"].lower() for m in user_msgs)

    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_fallback_on_exception(self, mock_comp):
        agent = _make_agent(config={})
        mock_comp.side_effect = RuntimeError("LLM down")

        messages = [{"role": "system", "content": "sys"}]
        result = agent._request_summary(messages)
        assert result == "Analysis failed to produce summary."

    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_fallback_on_empty(self, mock_comp):
        agent = _make_agent(config={})
        mock_comp.return_value = make_llm_response(content="")

        messages = [{"role": "system", "content": "sys"}]
        result = agent._request_summary(messages)
        assert result == "No summary generated."

    @patch("src.agents.analysis_code_agent.analysis_agent.completion")
    def test_messages_mutated(self, mock_comp):
        agent = _make_agent(config={})
        mock_comp.return_value = make_llm_response(content="<summary>Result</summary>")

        messages = [{"role": "system", "content": "sys"}]
        agent._request_summary(messages)
        # The method works on a copy; verify it actually called the LLM with more messages than given
        sent_messages = mock_comp.call_args.kwargs["messages"]
        assert len(sent_messages) > len(messages)


# ---------------------------------------------------------------------------
# R1.B: Baseline from current corpus
# ---------------------------------------------------------------------------


class TestBaselineFromCurrentCorpus:
    """R1.B — baseline must be computed dynamically, not from JSON file."""

    def test_agent_runner_has_compute_baseline(self):
        """AgentRunner has a _compute_baseline method."""
        ar_mod = _load_module("agent_runner_test", _AGENTS_DIR / "agent_runner.py")
        assert hasattr(ar_mod.AgentRunner, "_compute_baseline")

    def test_agent_runner_no_baseline_json_path(self):
        """AgentRunner no longer references _BASELINE_RESULTS_PATH."""
        ar_mod = _load_module("agent_runner_test2", _AGENTS_DIR / "agent_runner.py")
        assert not hasattr(ar_mod, "_BASELINE_RESULTS_PATH")


# ---------------------------------------------------------------------------
# Idea parsing (Phase 1 of generate_hypotheses_async)
# ---------------------------------------------------------------------------


# Fake LLM responses for idea parsing tests
_IDEAS_JSON = json.dumps([
    {
        "id": "H1",
        "description": "Chunk by sections",
        "rationale": "BM25 penalises long docs",
        "query_ids": "q1, q2",
        "falsifying": "recall does not improve",
    },
    {
        "id": "H2",
        "description": "Sliding window",
        "rationale": "Overlap preserves context",
        "query_ids": "q3",
        "falsifying": "IDF dilution",
    },
])

_IDEAS_JSON_RESPONSE = f"<hypotheses>{_IDEAS_JSON}</hypotheses>"

_IDEAS_MARKDOWN_RESPONSE = """\
### H1: Chunk by sections
Rationale: BM25 penalises long docs
Query IDs: q1, q2
Falsifying: recall does not improve

### H2: Sliding window
Rationale: Overlap preserves context
Query IDs: q3
Falsifying: IDF dilution
"""


def _make_code_agent(config=None):
    """Create a CodeAgent with default config, mocking system prompt read."""
    if config is None:
        config = {}
    with patch.object(pathlib.Path, "read_text", return_value="fake system prompt"):
        return CodeAgent(config)


class TestCodeAgentIdeaParsing:
    """Test that _generate_hypothesis_ideas parses both JSON and markdown formats."""

    @patch("src.agents.analysis_code_agent.code_agent.completion")
    def test_json_ideas_parsed(self, mock_comp):
        """LLM returns ideas in <hypotheses>JSON</hypotheses> — must parse correctly."""
        agent = _make_code_agent()
        mock_comp.return_value = make_llm_response(content=_IDEAS_JSON_RESPONSE)

        ideas = agent._generate_hypothesis_ideas(
            analysis_summary="summary", current_code="pass", n=4,
        )
        assert len(ideas) == 2
        assert ideas[0]["id"] == "H1"
        assert ideas[0]["description"] == "Chunk by sections"
        assert ideas[1]["id"] == "H2"

    @patch("src.agents.analysis_code_agent.code_agent.completion")
    def test_markdown_ideas_parsed(self, mock_comp):
        """LLM returns ideas in markdown format — must still work."""
        agent = _make_code_agent()
        mock_comp.return_value = make_llm_response(content=_IDEAS_MARKDOWN_RESPONSE)

        ideas = agent._generate_hypothesis_ideas(
            analysis_summary="summary", current_code="pass", n=4,
        )
        assert len(ideas) == 2
        assert ideas[0]["id"] == "H1"
        assert "sections" in ideas[0]["description"].lower()

    @patch("src.agents.analysis_code_agent.code_agent.completion")
    def test_unparseable_returns_empty(self, mock_comp):
        """LLM returns garbage — should return empty list, not crash."""
        agent = _make_code_agent()
        mock_comp.return_value = make_llm_response(content="I don't know what to do.")

        ideas = agent._generate_hypothesis_ideas(
            analysis_summary="summary", current_code="pass", n=4,
        )
        assert ideas == []

    @patch("src.agents.analysis_code_agent.code_agent.completion")
    def test_n_caps_returned_ideas(self, mock_comp):
        """Returned ideas are capped at n."""
        agent = _make_code_agent()
        mock_comp.return_value = make_llm_response(content=_IDEAS_JSON_RESPONSE)

        ideas = agent._generate_hypothesis_ideas(
            analysis_summary="summary", current_code="pass", n=1,
        )
        assert len(ideas) == 1

    @patch("src.agents.analysis_code_agent.llm_call.async_completion")
    @patch("src.agents.analysis_code_agent.code_agent.completion")
    def test_async_generates_hypotheses_from_json_ideas(self, mock_comp, mock_async_comp):
        """Full async path: JSON ideas parsed → Phase 2 code generation attempted."""
        import asyncio
        agent = _make_code_agent()

        # Phase 1 (sync completion): return JSON ideas
        mock_comp.return_value = make_llm_response(content=_IDEAS_JSON_RESPONSE)
        # Phase 2 (async_completion): return code for each idea
        code_resp = make_llm_response(
            content='### H1: test\n```python\nclass Preprocessor:\n    pass\n```'
        )

        async def fake_async_completion(**kwargs):
            return code_resp

        mock_async_comp.side_effect = fake_async_completion

        hypotheses = asyncio.run(agent.generate_hypotheses_async(
            analysis_summary="summary", current_code="pass", n=2,
        ))
        # Phase 1 parsed 2 ideas, Phase 2 called for each
        assert mock_comp.call_count == 1  # Phase 1
        assert mock_async_comp.call_count == 2  # Phase 2: one per idea
        assert len(hypotheses) == 2
