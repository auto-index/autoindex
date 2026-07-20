"""
run_tracker.py — Lightweight latency and token tracker for a single experiment run.

Both AnalysisAgent and CodeAgent accept a RunTracker and call record_llm_call()
after every LiteLLM completion. At end of run, to_dict() produces a JSON-serialisable
summary that is saved alongside the eval results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict


@dataclass
class LLMCallRecord:
    agent: str          # "analysis" | "code" | "one_shot"
    wall_time: float    # seconds for this call
    prompt_tokens: int
    completion_tokens: int


class RunTracker:
    def __init__(self) -> None:
        self._start = time.time()
        self._calls: list[LLMCallRecord] = []

    def record_llm_call(self, response, wall_time: float, agent: str = "") -> None:
        """Call immediately after a litellm.completion() returns."""
        usage = getattr(response, "usage", None)
        self._calls.append(LLMCallRecord(
            agent=agent,
            wall_time=wall_time,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        ))

    def to_dict(self) -> dict:
        total_wall = time.time() - self._start
        prompt = sum(c.prompt_tokens for c in self._calls)
        completion = sum(c.completion_tokens for c in self._calls)
        by_agent: dict[str, dict] = {}
        for c in self._calls:
            a = by_agent.setdefault(c.agent, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_time": 0.0})
            a["calls"] += 1
            a["prompt_tokens"] += c.prompt_tokens
            a["completion_tokens"] += c.completion_tokens
            a["wall_time"] += c.wall_time
        return {
            "total_wall_time_seconds": round(total_wall, 1),
            "llm_calls": len(self._calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "by_agent": by_agent,
        }
