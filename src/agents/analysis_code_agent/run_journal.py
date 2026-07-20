"""
run_journal.py — Per-run tracker for the AnalysisCodeAgent.

Records:
  - Per-iteration: recall@100, nDCG@10, which queries hit/missed
  - Per-hypothesis: targeted queries, delta scores, improved/regressed query IDs
  - Analysis: persistent failures, overfitting, convergence curve

Produces a structured summary string for feeding back into the LLM prompts.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class IterationRecord:
    iteration: int
    recall_at_100: float          # validation set
    ndcg_at_10: float             # validation set
    recall_at_10: float           # validation set
    n_queries: int
    hit_query_ids: list[str] = field(default_factory=list)
    miss_query_ids: list[str] = field(default_factory=list)
    adopted_hypothesis_id: Optional[str] = None
    eval_recall_at_100: Optional[float] = None   # held-out eval set
    eval_ndcg_at_10: Optional[float] = None      # held-out eval set
    eval_recall_at_10: Optional[float] = None    # held-out eval set


@dataclass
class HypothesisRecord:
    iteration: int
    h_id: str
    description: str
    rationale: str
    targeted_query_ids: list[str]
    delta_recall_100: float
    delta_recall_10: float
    delta_ndcg_10: float
    proven: bool
    adopted: bool
    improved_query_ids: list[str] = field(default_factory=list)
    regressed_query_ids: list[str] = field(default_factory=list)
    error: Optional[str] = None


class RunJournal:
    """Tracks the full history of an agent run for analysis and LLM feedback."""

    def __init__(self, run_dir: pathlib.Path) -> None:
        self._run_dir = run_dir
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._path = run_dir / "run_journal.json"
        self.iterations: list[IterationRecord] = []
        self.hypotheses: list[HypothesisRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_iteration(
        self,
        iteration: int,
        eval_results: dict,
        eval_results_harness: dict | None = None,
        adopted_hypothesis_id: Optional[str] = None,
    ) -> None:
        """Record the eval state at the start of a loop.

        eval_results: enriched val-set results from agent._enrich_eval_results()
        eval_results_harness: raw harness results from eval_queries (held-out set)
        """
        query_results = eval_results.get("query_results", [])
        hit_ids = [r["query_id"] for r in query_results if r.get("hit")]
        miss_ids = [r["query_id"] for r in query_results if not r.get("hit")]

        val_metrics = eval_results.get("metrics", {})
        eval_m = eval_results_harness.get("metrics", {}) if eval_results_harness else {}
        rec = IterationRecord(
            iteration=iteration,
            recall_at_100=val_metrics.get("recall_at_100", 0.0),
            ndcg_at_10=val_metrics.get("ndcg_at_10", 0.0),
            recall_at_10=val_metrics.get("recall_at_10", 0.0),
            n_queries=len(query_results),
            hit_query_ids=hit_ids,
            miss_query_ids=miss_ids,
            adopted_hypothesis_id=adopted_hypothesis_id,
            eval_recall_at_100=eval_m.get("recall_at_100"),
            eval_ndcg_at_10=eval_m.get("ndcg_at_10"),
            eval_recall_at_10=eval_m.get("recall_at_10"),
        )
        self.iterations.append(rec)
        self.save()

    def record_hypothesis(
        self,
        iteration: int,
        h_id: str,
        description: str,
        rationale: str,
        targeted_query_ids: list[str],
        delta_recall_100: float,
        delta_recall_10: float,
        delta_ndcg_10: float,
        proven: bool,
        adopted: bool,
        improved_query_ids: list[str],
        regressed_query_ids: list[str],
        error: Optional[str] = None,
    ) -> None:
        rec = HypothesisRecord(
            iteration=iteration,
            h_id=h_id,
            description=description,
            rationale=rationale,
            targeted_query_ids=targeted_query_ids,
            delta_recall_100=delta_recall_100,
            delta_recall_10=delta_recall_10,
            delta_ndcg_10=delta_ndcg_10,
            proven=proven,
            adopted=adopted,
            improved_query_ids=improved_query_ids,
            regressed_query_ids=regressed_query_ids,
            error=error,
        )
        self.hypotheses.append(rec)
        self.save()

    def set_iteration_adoption(
        self,
        iteration: int,
        adopted_hypothesis_id: Optional[str],
    ) -> None:
        """Update adopted hypothesis for a previously recorded iteration."""
        for rec in self.iterations:
            if rec.iteration == iteration:
                rec.adopted_hypothesis_id = adopted_hypothesis_id
                self.save()
                return

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def persistent_failure_ids(self, min_iters: int = 2) -> list[str]:
        """Query IDs that failed (not hit@100) in at least min_iters iterations."""
        if not self.iterations:
            return []
        # Count how many iterations each query missed
        miss_count: dict[str, int] = {}
        for rec in self.iterations:
            for qid in rec.miss_query_ids:
                miss_count[qid] = miss_count.get(qid, 0) + 1
        cutoff = min(min_iters, len(self.iterations))
        return [qid for qid, cnt in miss_count.items() if cnt >= cutoff]

    def overfitting_cases(self) -> list[HypothesisRecord]:
        """Adopted hypotheses where regressions outnumber improvements."""
        return [
            h for h in self.hypotheses
            if h.adopted
            and len(h.regressed_query_ids) > len(h.improved_query_ids)
        ]

    def convergence_curve(self) -> list[dict]:
        return [
            {
                "iteration": r.iteration,
                "recall_at_100": r.recall_at_100,
                "ndcg_at_10": r.ndcg_at_10,
                "adopted": r.adopted_hypothesis_id,
            }
            for r in self.iterations
        ]

    def hypothesis_win_rate(self) -> dict:
        total = len(self.hypotheses)
        if total == 0:
            return {"total": 0, "proven": 0, "adopted": 0, "win_rate": 0.0}
        proven = sum(1 for h in self.hypotheses if h.proven)
        adopted = sum(1 for h in self.hypotheses if h.adopted)
        return {
            "total": total,
            "proven": proven,
            "adopted": adopted,
            "win_rate": proven / total,
        }

    # ------------------------------------------------------------------
    # Summary for LLM prompt
    # ------------------------------------------------------------------

    def summary_for_prompt(self) -> str:
        """Return a concise structured summary to include in analysis/code agent prompts."""
        if not self.iterations:
            return "(No run history yet.)"

        lines: list[str] = ["## Run Journal"]

        # Convergence curve
        lines.append("\n### Score history")
        prev_recall = None
        for r in self.iterations:
            delta = f"  ({r.recall_at_100 - prev_recall:+.4f} recall)" if prev_recall is not None else "  (first)"
            adopted_note = f"  [adopted {r.adopted_hypothesis_id}]" if r.adopted_hypothesis_id else ""
            lines.append(
                f"  iter {r.iteration}: recall@100={r.recall_at_100:.4f}  "
                f"nDCG@10={r.ndcg_at_10:.4f}{delta}{adopted_note}"
            )
            prev_recall = r.recall_at_100

        # NOTE: persistent-failure listings have been intentionally removed from
        # the journal summary. In prior runs the explicit "prioritise these"
        # instruction caused the analysis agent to fixate on the same 3-4
        # queries every iteration and produce narrative-driven, non-generalising
        # recommendations. Most "persistent" failures were unfixable BM25-side
        # vocabulary gaps anyway. The agent now decides which queries to look
        # at from the per-iteration analysis-targets list.

        # Overfitting
        ov = self.overfitting_cases()
        if ov:
            lines.append(f"\n### Overfitting detected ({len(ov)} adopted hypothesis/es hurt more queries than helped)")
            for h in ov:
                lines.append(
                    f"  - {h.h_id} (iter {h.iteration}): \"{h.description}\"\n"
                    f"    improved {len(h.improved_query_ids)}, "
                    f"regressed {len(h.regressed_query_ids)}: {', '.join(h.regressed_query_ids[:10])}"
                )
            lines.append(
                "  → Future hypotheses should explicitly avoid regressing these query IDs."
            )

        # Recent hypothesis outcomes
        if self.hypotheses:
            win = self.hypothesis_win_rate()
            lines.append(
                f"\n### Hypothesis history  "
                f"(win rate: {win['proven']}/{win['total']} proven, "
                f"{win['adopted']} adopted)"
            )
            for h in self.hypotheses[-8:]:  # last 8
                status = "✓ adopted" if h.adopted else ("✓ proven" if h.proven else "✗ failed")
                lines.append(
                    f"  iter {h.iteration} | {h.h_id}: {h.description[:80]}\n"
                    f"    {status}  Δrecall={h.delta_recall_100:+.4f}  ΔnDCG={h.delta_ndcg_10:+.4f}"
                    + (f"  improved={len(h.improved_query_ids)} regressed={len(h.regressed_query_ids)}" if not h.error else f"  error={h.error[:60]}")
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        data = {
            "iterations": [asdict(r) for r in self.iterations],
            "hypotheses": [asdict(r) for r in self.hypotheses],
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: pathlib.Path) -> "RunJournal":
        journal = cls(run_dir)
        path = run_dir / "run_journal.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            valid_fields = {f.name for f in IterationRecord.__dataclass_fields__.values()}
            journal.iterations = [
                IterationRecord(**{k: v for k, v in r.items() if k in valid_fields})
                for r in data.get("iterations", [])
            ]
            journal.hypotheses = [HypothesisRecord(**r) for r in data.get("hypotheses", [])]
        return journal
