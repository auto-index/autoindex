"""
analysis_agent.py - Multi-turn tool-calling analysis agent.

Uses LiteLLM to call a model that can invoke BM25/file tools to investigate
BM25 retrieval failures and produce a structured analysis summary.
"""
from __future__ import annotations

import json
import logging
import os
import re
import pathlib
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from .llm_call import completion

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent

logger = logging.getLogger("analysis_code_agent")

import re as _re
import sys as _sys
_sys.path.insert(0, str(_AGENT_DIR))
from analysis_tools.tools import TOOL_SCHEMAS, dispatch_tool


def load_corpus_description(split: str) -> str:
    """Return the corpus description markdown for *split*.

    Lookup order:
    1. ``context/corpus_descriptions/{split}.md``
    2. ``context/corpus_descriptions/{split minus _Ndocs suffix}.md``
    3. ``context/corpus_descriptions/tip_of_the_tongue.md``  (fallback)
    """
    corpus_desc_dir = _AGENT_DIR / "context" / "corpus_descriptions"
    path = corpus_desc_dir / f"{split}.md"
    if not path.exists():
        base_split = _re.sub(r"_\d+docs$", "", split)
        path = corpus_desc_dir / f"{base_split}.md"
    if not path.exists():
        path = corpus_desc_dir / "tip_of_the_tongue.md"
    return path.read_text(encoding="utf-8")


@dataclass
class AnalysisResult:
    summary: str
    turns: int
    conversation: list[dict]  # full message history for logging


class AnalysisAgent:
    def __init__(
        self,
        config: dict,
        tracker=None,
        split: str = "tip_of_the_tongue",
        n_val_queries: int = 0,
        n_eval_queries: int = 0,
    ) -> None:
        self._tracker = tracker
        self._model = config.get("analysis_model", "openai/gpt-4o-mini")
        self._temperature = config.get("analysis_temperature", 0.3)
        self._max_turns = config.get("analysis_max_turns", 8)
        self._min_tool_turns = config.get("min_tool_turns", 3)
        self._use_tools = config.get("use_tools", True)
        self._bash_timeout = config.get("bash_timeout_seconds", 30)
        self._llm_timeout = config.get("analysis_llm_timeout", None)
        self._n_failures_shown = config.get("n_failures_shown", 5)
        self._n_hard_negatives_shown = config.get("n_hard_negatives_shown", 5)
        self._n_successes_shown = config.get("n_successes_shown", 8)
        # Only pass api_key explicitly for proxy; native providers read key from env.
        _proxy_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LITE_LLM_KEY")
            or os.environ.get("LITELLM_API_KEY")
            or ""
        )
        self._api_key = _proxy_key if config.get("api_base") else None
        self._api_base = config.get("api_base", "https://thekeymaker.umass.edu/")

        # Load system prompt, injecting per-split corpus description and concrete query counts
        system_path = _AGENT_DIR / "context" / "ANALYSIS_SYSTEM.md"
        template = system_path.read_text(encoding="utf-8")
        one_query_pct = (100.0 / n_val_queries) if n_val_queries else 0.0
        self._system_prompt = (
            template
            .replace("{{CORPUS_DESCRIPTION}}", load_corpus_description(split))
            .replace("{{VAL_QUERY_COUNT}}", str(n_val_queries))
            .replace("{{EVAL_QUERY_COUNT}}", str(n_eval_queries))
            .replace("{{VAL_ONE_QUERY_PCT}}", f"+{one_query_pct:.2f}%")
        )

    def analyze(
        self,
        eval_results: dict,
        baseline_results: dict,
        current_code: str,
        client,
        split: str = "tip_of_the_tongue",
        journal_summary: str | None = None,
    ) -> AnalysisResult:
        """Run multi-turn analysis loop. Returns AnalysisResult with summary."""

        # Build candidate analysis targets
        candidates = self._build_candidates(eval_results, baseline_results)

        # Build initial user message
        initial_msg = self._build_initial_context(
            eval_results=eval_results,
            baseline_results=baseline_results,
            current_code=current_code,
            candidates=candidates,
            split=split,
            journal_summary=journal_summary,
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": initial_msg},
        ]

        tool_turns = 0
        summary_text = None

        for turn in range(self._max_turns):
            msg = self._call_llm(messages, turn, tools=TOOL_SCHEMAS if self._use_tools else None)
            if msg is None:
                break

            # Append assistant message as dict (preserving tool_calls)
            tool_calls = getattr(msg, "tool_calls", None) or []
            assistant_dict = {"role": "assistant", "content": msg.content}
            if tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_dict)

            # If tool calls present, dispatch them
            if tool_calls:
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse tool args as JSON for %s. Raw: %r",
                            tc.function.name, tc.function.arguments,
                        )
                        args = {}
                    try:
                        result = dispatch_tool(tc.function.name, args, client=client, split=split)
                        logger.debug(
                            "Tool call %s(%s) → %d chars",
                            tc.function.name, args, len(str(result)),
                        )
                    except Exception as e:
                        logger.exception(
                            "Tool call %s(%s) raised",
                            tc.function.name, args,
                        )
                        result = f"[tool error] {type(e).__name__}: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                tool_turns += 1
                continue

            # No tool calls — check for <summary> tag
            text = msg.content or ""
            match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
            if match:
                summary_text = match.group(1).strip()
                break

            # No summary tag — nudge if not enough tool turns (only when tools are enabled)
            if self._use_tools and tool_turns < self._min_tool_turns:
                nudge = (
                    f"You have only completed {tool_turns}/{self._min_tool_turns} tool-using turns. "
                    f"Please investigate more failing queries using the available tools "
                    f"(bm25_retrieve, read_file, grep_search) before providing your summary."
                )
                messages.append({"role": "user", "content": nudge})
                continue

            # No summary tag, no tool calls — exit
            break

        # If no summary found in loop, check all assistant messages for <summary>
        if summary_text is None:
            for m in messages:
                if m.get("role") == "assistant" and m.get("content"):
                    match = re.search(r"<summary>(.*?)</summary>", m["content"], re.DOTALL)
                    if match:
                        summary_text = match.group(1).strip()
                        break

        # If still no summary, request one
        if summary_text is None:
            summary_text = self._request_summary(messages)

        return AnalysisResult(
            summary=summary_text,
            turns=len([m for m in messages if m.get("role") == "assistant"]),
            conversation=messages,
        )

    def _call_llm(self, messages: list[dict], turn: int, tools=None):
        """Call LLM with retry logic. Returns response message object or None."""
        def _do_call(msgs):
            kwargs = dict(
                model=self._model,
                messages=msgs,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
                timeout=self._llm_timeout,
            )
            if tools:
                kwargs["tools"] = tools
            t0 = time.time()
            response = completion(**kwargs)
            if self._tracker:
                self._tracker.record_llm_call(response, time.time() - t0, agent="analysis")
            return response.choices[0].message

        try:
            return _do_call(messages)
        except Exception as e:
            logger.exception("Analysis LLM call failed on turn %d", turn)
            print(f"[analysis_agent] LLM call failed (turn {turn}): {type(e).__name__}: {e}")
            if "ContentPolicyViolation" in type(e).__name__ or "content_policy" in str(e).lower() or "content management policy" in str(e).lower():
                sanitized = self._sanitize_messages(messages)
                messages[:] = sanitized
                try:
                    return _do_call(sanitized)
                except Exception as e2:
                    logger.exception("Analysis LLM retry (sanitized) failed on turn %d", turn)
                    print(f"[analysis_agent] LLM retry (sanitized) failed: {type(e2).__name__}: {e2}")
                    return None
            time.sleep(5)
            try:
                return _do_call(messages)
            except Exception as e2:
                logger.exception("Analysis LLM retry failed on turn %d", turn)
                print(f"[analysis_agent] LLM retry failed: {type(e2).__name__}: {e2}")
                return None

    def _sanitize_messages(self, messages: list[dict]) -> list[dict]:
        """Return messages with tool results truncated to avoid content policy violations."""
        sanitized = []
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content", "")
                if len(content) > 400:
                    content = content[:400] + "\n... [truncated]"
                sanitized.append({**m, "content": content})
            else:
                sanitized.append(m)
        return sanitized

    def _strip_tool_messages(self, messages: list[dict]) -> list[dict]:
        """Remove tool_calls from assistant messages and tool-role messages.

        This lets us send the conversation to providers (e.g. Bedrock) that
        reject tool_calls/tool messages when no ``tools`` param is given.
        """
        cleaned = []
        for m in messages:
            if m.get("role") == "tool":
                # Convert tool result to a user message so context is preserved
                cleaned.append({
                    "role": "user",
                    "content": f"[Tool result for {m.get('tool_call_id', '?')}]: {m.get('content', '')}",
                })
            elif m.get("role") == "assistant" and m.get("tool_calls"):
                # Keep assistant text but drop tool_calls
                content = m.get("content") or ""
                tool_summaries = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_summaries.append(f"[Called {fn.get('name', '?')}]")
                combined = (content + "\n" + "\n".join(tool_summaries)).strip()
                cleaned.append({"role": "assistant", "content": combined})
            else:
                cleaned.append(m)
        return cleaned

    def _request_summary(self, messages: list[dict]) -> str:
        """Ask the LLM for a final summary when none was produced in the loop."""
        # Strip tool-related messages so providers that don't support tools= work
        summary_messages = self._strip_tool_messages(messages)
        summary_messages.append({
            "role": "user",
            "content": (
                "IMPORTANT: All tools have been disabled. You CANNOT make any more tool calls.\n\n"
                "Based on everything you have investigated so far, provide your FINAL analysis summary NOW.\n"
                "Provide a structured summary of failure patterns and recommendations "
                "with concrete evidence (query IDs/doc IDs) from your earlier investigation.\n\n"
                "You MUST wrap your entire analysis in <summary>...</summary> tags. "
                "Do NOT attempt to call any tools — just write the summary."
            ),
        })
        try:
            response = completion(
                model=self._model,
                messages=summary_messages,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
                timeout=self._llm_timeout,
            )
            msg = response.choices[0].message
            text = msg.content or ""

            # Guardrail: if LLM produced tool calls or no <summary> tags, retry once
            needs_retry = (getattr(msg, "tool_calls", None) or False) or "<summary>" not in text
            if needs_retry:
                summary_messages.append({"role": "assistant", "content": text})
                summary_messages.append({
                    "role": "user",
                    "content": (
                        "You did NOT produce a summary. Tools are disabled — do not attempt tool calls.\n"
                        "Write your final analysis summary NOW based on what you already investigated. "
                        "Wrap it in <summary>...</summary> tags."
                    ),
                })
                response2 = completion(
                    model=self._model,
                    messages=summary_messages,
                    temperature=self._temperature,
                    api_key=self._api_key,
                    api_base=self._api_base,
                    timeout=self._llm_timeout,
                )
                text = response2.choices[0].message.content or "No summary generated."

            # Extract <summary> if present
            match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
            summary = match.group(1).strip() if match else text

            if not summary:
                summary = "No summary generated."

            return summary
        except Exception as e:
            logger.exception("Summary request failed")
            print(f"[analysis_agent] _request_summary failed: {type(e).__name__}: {e}")
            return "Analysis failed to produce summary."

    def _build_candidates(self, eval_results: dict, baseline_results: dict) -> dict:
        """Build candidate analysis targets from eval results."""
        query_results = eval_results.get("query_results", [])
        baseline_qr = {
            r["query_id"]: r for r in baseline_results.get("query_results", [])
        }

        # Failures (regressions): baseline had hit but current doesn't
        failures = []
        for r in query_results:
            baseline_r = baseline_qr.get(r["query_id"])
            if baseline_r and baseline_r.get("hit") and not r.get("hit"):
                failures.append(r)

        failures = failures[: self._n_failures_shown]

        # Hard negatives: missed queries, top-10 retrieved that aren't gold
        misses = [r for r in query_results if not r.get("hit")]
        hard_negatives = []
        for r in misses[: self._n_hard_negatives_shown]:
            wrong_docs = [
                doc_id
                for doc_id in r.get("retrieved_doc_ids", [])[:10]
                if doc_id not in r.get("relevant_doc_ids", [])
            ][:3]
            if wrong_docs:
                hard_negatives.append({
                    "query_id": r["query_id"],
                    "query_text": r.get("query_text", ""),
                    "relevant_doc_ids": r.get("relevant_doc_ids", []),
                    "wrong_docs": wrong_docs,
                })

        # Successes: queries with hits, sorted by worst rank first
        hits = [r for r in query_results if r.get("hit")]
        successes = sorted(
            hits, key=lambda x: x.get("rank") or 0, reverse=True
        )[: self._n_successes_shown]

        return {
            "failures": failures,
            "hard_negatives": hard_negatives,
            "successes": successes,
        }

    def _build_initial_context(
        self,
        eval_results: dict,
        baseline_results: dict,
        current_code: str,
        candidates: dict,
        split: str = "tip_of_the_tongue",
        journal_summary: str | None = None,
    ) -> str:
        """Build the initial user message with all context."""

        failures = candidates["failures"]
        hard_negatives = candidates["hard_negatives"]
        successes = candidates["successes"]

        # Format failures section
        if failures:
            lines = []
            for r in failures:
                lines.append(
                    f"  - [{r['query_id']}] \"{r.get('query_text', 'N/A')}\""
                )
                lines.append(f"    Expected: {r.get('relevant_doc_ids', [])}")
                lines.append(
                    f"    Retrieved: {r.get('retrieved_doc_ids', [])[:5]}"
                )
            failures_text = (
                f"### Regressions ({len(failures)} queries "
                f"-- baseline hit, current missed):\n" + "\n".join(lines)
            )
        else:
            failures_text = "### Regressions: none"

        # Format hard negatives
        if hard_negatives:
            lines = []
            for hn in hard_negatives:
                lines.append(
                    f"  - [{hn['query_id']}] \"{hn['query_text']}\""
                )
                lines.append(
                    f"    Gold: {hn['relevant_doc_ids']}, "
                    f"Wrong top docs: {hn['wrong_docs']}"
                )
            hn_text = (
                f"### Hard negatives ({len(hard_negatives)} queries):\n"
                + "\n".join(lines)
            )
        else:
            hn_text = "### Hard negatives: none"

        # Format successes
        if successes:
            lines = [
                f"  - [{r['query_id']}] rank={r.get('rank', 'N/A')} "
                f"\"{r.get('query_text', 'N/A')}\""
                for r in successes
            ]
            succ_text = (
                f"### Successes (hit, worst rank first, "
                f"{len(successes)} shown):\n" + "\n".join(lines)
            )
        else:
            succ_text = "### Successes: none"

        # Get metrics
        metrics = eval_results.get("metrics", {})
        recall_100 = metrics.get(
            "recall_at_100", eval_results.get("recall_at_k", 0)
        )
        ndcg_10 = metrics.get("ndcg_at_10", eval_results.get("ndcg", 0))
        baseline_recall = baseline_results.get("recall_at_k", 0)
        baseline_ndcg = baseline_results.get("ndcg", 0)

        journal_section = f"\n{journal_summary}\n" if journal_summary else ""

        tool_section = (
            f"## Available Tools\n"
            f"You have three tools available (invoked via the tool-calling API, NOT via text):\n\n"
            f"1. **bm25_retrieve(query, top_k=10)** — "
            f"Query the current BM25 index. Returns doc_id, score, rank for each result.\n"
            f"2. **read_file(file_path, max_chars=800, filter_id=None)** — "
            f"Read a file from data/{split}/. file_path is relative (e.g. \"documents.jsonl\" or \"validation_queries.jsonl\"). "
            f"Use filter_id to look up a specific doc_id or query_id in JSONL files.\n"
            f"3. **grep_search(pattern, file_path, max_results=10)** — "
            f"Regex search within a data file. file_path is relative to data/{split}/.\n\n"
            f"Investigate the failures and patterns above using these tools.\n"
            f"When done investigating, wrap your final analysis in <summary>...</summary> tags.\n"
        )

        n_val = len(eval_results.get("query_results", []))
        return (
            f"{journal_section}"
            f"## Current Evaluation (validation set — {n_val} queries)\n"
            f"> Note: these metrics are on a small validation set. Hypotheses adopted here will be\n"
            f"> tested on a separate held-out eval set that is never used to guide\n"
            f"> decisions. Prioritise generalizable strategies over fixes for specific val queries.\n"
            f"\n"
            f"- Recall@100: {recall_100:.4f} (baseline: {baseline_recall:.4f})\n"
            f"- nDCG@10: {ndcg_10:.4f} (baseline: {baseline_ndcg:.4f})\n"
            f"\n"
            f"## Current preprocess.py\n"
            f"```python\n"
            f"{current_code}\n"
            f"```\n"
            f"\n"
            f"## Analysis Targets\n"
            f"You are required to investigate BOTH failures AND successes — failures show what's broken,\n"
            f"successes show what signal the index is currently exploiting (which any change must preserve).\n"
            f"\n"
            f"{failures_text}\n"
            f"\n"
            f"{hn_text}\n"
            f"\n"
            f"{succ_text}\n"
            f"\n"
            f"{tool_section}"
        )
