"""
agent.py - AnalysisCodeAgent: two-stage analysis + hypothesis-testing orchestrator.

Overrides AgentRunner.run() to implement:
1. Analysis agent investigates failures via multi-turn bash loop
2. Code agent generates hypotheses, tests each on BM25 server, synthesizes final code
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import random

import subprocess
import sys
import time
import pathlib
import datetime

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent

# Add evaluation to path
_EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from ..agent_runner import AgentRunner
from .analysis_agent import AnalysisAgent
from .code_agent import CodeAgent
from .bm25_client import BM25Client
from .run_journal import RunJournal
from .run_tracker import RunTracker


_DEBUG_LOGGER_NAME = "analysis_code_agent"
logger = logging.getLogger(_DEBUG_LOGGER_NAME)


def _setup_debug_logger(experiment_dir: pathlib.Path) -> logging.Logger:
    """Configure a DEBUG-level FileHandler at {experiment_dir}/debug.log.

    Idempotent: safe to call multiple times per process (clears previous
    per-run handlers so each run writes to its own experiment directory).
    """
    log = logging.getLogger(_DEBUG_LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    fh = logging.FileHandler(experiment_dir / "debug.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(fh)
    log.info("Debug log initialized at %s", experiment_dir / "debug.log")
    return log


def _load_config(overrides: dict | None = None) -> dict:
    config_path = _AGENT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def _load_data(split: str = "tip_of_the_tongue", corpus_size: int | None = None, seed: int = 42):
    """Load queries and corpus from data/.

    If corpus_size is None, loads every document. Otherwise uses reservoir sampling
    to select corpus_size documents, always retaining every gold doc (any doc
    referenced by at least one query's relevant_doc_ids).
    """
    import random
    from schema import Document, EvalQuery

    data_dir = _PROJECT_ROOT / "data" / split
    if not data_dir.exists():
        data_dir = _PROJECT_ROOT / "data"

    def _load_queries_file(queries_path) -> list[EvalQuery]:
        queries = []
        if queries_path.exists():
            with queries_path.open(encoding="utf-8") as f:
                for line in f:
                    q = json.loads(line)
                    queries.append(EvalQuery(
                        query_id=q["query_id"],
                        query_text=q.get("query_text") or q.get("query_content", ""),
                        relevant_doc_ids=q["relevant_doc_ids"],
                    ))
        return queries

    val_queries = _load_queries_file(data_dir / "validation_queries.jsonl")
    eval_queries = _load_queries_file(data_dir / "evaluation_queries.jsonl")
    if not val_queries and not eval_queries:
        val_queries = _load_queries_file(data_dir / "queries.jsonl")
        eval_queries = val_queries

    docs_path = data_dir / "documents.jsonl"
    gold_doc_ids = {doc_id for q in val_queries + eval_queries for doc_id in q.relevant_doc_ids}
    target_non_gold = max(0, corpus_size - len(gold_doc_ids)) if corpus_size is not None else None

    gold_docs: list[Document] = []
    reservoir: list[Document] = []
    rng = random.Random(seed)
    n_non_gold_seen = 0

    with docs_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc = Document(doc_id=d["doc_id"], text=d["text"], metadata=d.get("metadata", {}))
            if doc.doc_id in gold_doc_ids:
                gold_docs.append(doc)
            elif target_non_gold is None:
                reservoir.append(doc)
            else:
                if n_non_gold_seen < target_non_gold:
                    reservoir.append(doc)
                else:
                    j = rng.randint(0, n_non_gold_seen)
                    if j < target_non_gold:
                        reservoir[j] = doc
                n_non_gold_seen += 1

    docs = gold_docs + reservoir
    print(f"[data] Corpus: {len(docs)} docs ({len(gold_docs)} gold + {len(reservoir)} non-gold), "
          f"{len(val_queries)} val queries, {len(eval_queries)} eval queries")
    return docs, val_queries, eval_queries


class AnalysisCodeAgent(AgentRunner):
    agent_name = "analysis_code_agent"

    def __init__(self, use_history: bool = True, use_contrastive: bool = True, use_analysis: bool = True, model: str | None = None, api_base: str | None = None) -> None:
        overrides: dict = {}
        if model:
            overrides["analysis_model"] = model
            overrides["code_model"] = model
            # If model overridden but no explicit api_base, clear it so LiteLLM
            # routes to the provider's native endpoint (e.g. Google for gemini/).
            overrides["api_base"] = api_base  # None = native endpoint
        elif api_base is not None:
            overrides["api_base"] = api_base
        self._config = _load_config(overrides or None)
        self._use_history = use_history
        self._use_contrastive = use_contrastive
        self._use_analysis = use_analysis
        self._server_process = None
        self._client = BM25Client(
            base_url=f"http://localhost:{self._config.get('server_port', 8765)}",
            batch_size=self._config.get("bm25_batch_size", 100_000),
        )
        # Set by run() after data loading; used by run_eval()
        self._documents: list | None = None
        self._queries: list | None = None

    # --- AgentRunner ABC stubs (we override run() instead) ---

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        return ""  # not used

    def call_llm(self, prompt: str, iteration: int) -> None:
        pass  # not used

    # --- Eval: use sampled corpus via BM25 server (not the full 1M-doc harness) ---

    def run_eval(self, iteration: int | None = None, queries: list | None = None) -> dict:
        """Evaluate current preprocess.py against the sampled corpus via BM25 server.

        Overrides AgentRunner.run_eval() to avoid reloading 1M docs from disk.
        Builds a 'harness_eval' index on the server and runs all queries.
        """
        from .eval_utils import load_preprocessor_from_code, run_subset_eval, sanitize_docs_for_preprocessing, remap_chunk_doc_ids

        eval_queries = queries if queries is not None else self._queries

        if self._documents is None or eval_queries is None:
            raise RuntimeError("run_eval() called before documents/queries were loaded.")

        preprocess_path = _AGENT_DIR / "preprocess.py"
        code = preprocess_path.read_text(encoding="utf-8")
        preprocessor = load_preprocessor_from_code(code)

        print(f"[agent] Preprocessing {len(self._documents)} documents ...")
        sanitized_docs, reverse_map = sanitize_docs_for_preprocessing(self._documents)
        chunks = preprocessor.preprocess(sanitized_docs)
        remap_chunk_doc_ids(chunks, reverse_map)
        print(f"[agent] Built {len(chunks)} chunks. Pushing to BM25 server ...")
        self._client.build_index("harness_eval", chunks, persist=False)

        summary = run_subset_eval("harness_eval", eval_queries, self._client, top_k=100)

        agent_name = getattr(preprocessor, "name", type(preprocessor).__name__)
        iter_str = f" (Iteration {iteration})" if iteration is not None else ""
        print(
            f"\n{'='*60}\n"
            f"Agent       : {agent_name}{iter_str}\n"
            f"{'='*60}\n"
            f"  Recall@10  : {summary.recall_at_10:.4f}\n"
            f"  Recall@100 : {summary.recall_at_100:.4f}\n"
            f"  nDCG@10    : {summary.ndcg_at_10:.4f}\n"
        )

        return {
            "agent": agent_name,
            "config": {
                "top_k": 100,
                "n_docs": len(self._documents),
                "n_queries": len(eval_queries),
                "n_chunks": len(chunks),
                "chunks_per_doc": len(chunks) / max(len(self._documents), 1),
            },
            "metrics": {
                "recall_at_10": summary.recall_at_10,
                "recall_at_100": summary.recall_at_100,
                "ndcg_at_10": summary.ndcg_at_10,
            },
            "query_results": [],  # enriched separately by _enrich_eval_results
        }

    # --- Server management ---

    def _ensure_server_running(self) -> None:
        """Start the BM25 FastAPI server if not already up."""
        if self._client.health():
            print("[agent] BM25 server already running.")
            return

        port = self._config.get("server_port", 8765)
        persist_dir = self._config.get("server_persist_dir", ".bm25_cache")
        server_path = _AGENT_DIR / "bm25_server.py"

        print(f"[agent] Starting BM25 server on port {port} ...")
        self._server_process = subprocess.Popen(
            ["uv", "run", "python", str(server_path), "--port", str(port), "--persist-dir", persist_dir],
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self._kill_server)

        # Wait for server to be ready; configurable so small-corpus runs aren't penalised
        max_wait = self._config.get("server_startup_timeout", 30)
        for i in range(max_wait * 2):
            time.sleep(0.5)
            if self._client.health():
                print(f"[agent] BM25 server ready (took {(i+1)*0.5:.1f}s).")
                return

        raise RuntimeError(
            f"BM25 server failed to start after {max_wait}s. "
            f"Check: uv run python {server_path} --port {port}"
        )

    def _kill_server(self) -> None:
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            print("[agent] BM25 server stopped.")

    # --- Per-query enrichment ---

    def _enrich_eval_results(
        self, raw_results: dict, queries: list, client
    ) -> dict:
        """
        The eval harness returns metrics but no per-query hit/rank data.
        Enrich raw_results with query_results using the BM25 server's 'current' index.
        """
        from .eval_utils import run_subset_eval

        eval_summary = run_subset_eval("current", queries, client, top_k=100)

        query_results = []
        for pq, q in zip(eval_summary.per_query, queries):
            query_results.append({
                "query_id": pq.query_id,
                "query_text": q.query_text,
                "hit": pq.hit_at_100,
                "rank": pq.rank,
                "relevant_doc_ids": q.relevant_doc_ids,
                "retrieved_doc_ids": pq.retrieved_doc_ids,
            })

        raw_results["query_results"] = query_results
        return raw_results

    # --- Preprocessing helper ---

    def _compute_baseline(self, queries: list | None = None) -> dict:
        """Run baseline preprocessor on the current sampled corpus via BM25 server.

        Overrides AgentRunner._compute_baseline() so that baseline results are
        computed on the exact same document set as the current eval (R1.B).
        """
        from .eval_utils import run_subset_eval

        _EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"
        _AGENTS_DIR = _PROJECT_ROOT / "src" / "agents"
        for p in [str(_EVAL_DIR), str(_AGENTS_DIR)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from baseline.preprocess import Preprocessor as BaselinePreprocessor

        eval_queries = queries if queries is not None else self._queries

        if self._documents is None or eval_queries is None:
            raise RuntimeError("_compute_baseline() called before documents/queries were loaded.")

        baseline_preprocessor = BaselinePreprocessor()
        baseline_chunks = baseline_preprocessor.preprocess(self._documents)
        print(f"[agent] Baseline: {len(baseline_chunks)} chunks from {len(self._documents)} docs")
        self._client.build_index("baseline", baseline_chunks, persist=False)

        baseline_summary = run_subset_eval("baseline", eval_queries, self._client, top_k=100)
        return {
            "recall_at_k": baseline_summary.recall_at_100,
            "ndcg": baseline_summary.ndcg_at_10,
            "query_results": [
                {
                    "query_id": pq.query_id,
                    "hit": pq.hit_at_100,
                    "rank": pq.rank,
                    "retrieved_doc_ids": pq.retrieved_doc_ids,
                }
                for pq in baseline_summary.per_query
            ],
        }

    def _preprocess_with_current_code(self, documents: list, current_code: str) -> list:
        """Load preprocessor from current code and run it on documents."""
        from .eval_utils import load_preprocessor_from_code, sanitize_docs_for_preprocessing, remap_chunk_doc_ids
        preprocessor = load_preprocessor_from_code(current_code)
        sanitized_docs, reverse_map = sanitize_docs_for_preprocessing(documents)
        chunks = preprocessor.preprocess(sanitized_docs)
        return remap_chunk_doc_ids(chunks, reverse_map)

    # --- Logging ---

    def _log_analysis(self, iteration: int, analysis_result) -> None:
        logs_dir = self._experiment_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # Full conversation log
        log_path = logs_dir / f"iteration_{iteration}_analysis.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"=== Analysis Agent | Iteration {iteration} | {timestamp} ===\n\n")

            for i, msg in enumerate(analysis_result.conversation):
                role = msg.get("role", "unknown").upper()
                f.write(f"--- MESSAGE {i} [{role}] ---\n")

                if msg.get("content"):
                    f.write(f"{msg['content']}\n")

                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        f.write(f"[TOOL CALL] id={tc.get('id')} name={fn.get('name')} args={fn.get('arguments')}\n")

                if role == "TOOL":
                    f.write(f"[TOOL RESULT] tool_call_id={msg.get('tool_call_id')}\n")

                f.write("\n")

            f.write("--- FINAL SUMMARY ---\n")
            f.write(analysis_result.summary)

        # Summary-only file
        summary_path = logs_dir / f"iteration_{iteration}_analysis_summary.txt"
        summary_path.write_text(analysis_result.summary, encoding="utf-8")

        print(f"[agent] Analysis log: {log_path}")

    def _log_hypotheses(self, iteration: int, hypothesis_results: list) -> None:
        logs_dir = self._experiment_dir
        logs_dir.mkdir(parents=True, exist_ok=True)

        data = []
        for r in hypothesis_results:
            h = r.hypothesis
            data.append({
                "id": h.id,
                "description": h.description,
                "rationale": h.rationale,
                "code": h.code,
                "query_ids_to_test": h.query_ids_to_test,
                "falsifying_condition": h.falsifying_condition,
                "test_results": {
                    "hypothesis_recall_100": r.hypothesis_recall_100,
                    "baseline_recall_100": r.baseline_recall_100,
                    "delta_recall_100": r.delta_recall_100,
                    "hypothesis_recall_10": r.hypothesis_recall_10,
                    "baseline_recall_10": r.baseline_recall_10,
                    "delta_recall_10": r.delta_recall_10,
                    "delta_ndcg_10": r.delta_ndcg_10,
                    "proven": r.proven,
                    "notes": r.notes,
                    "error": r.error,
                },
            })

        log_path = logs_dir / f"iteration_{iteration}_hypotheses.json"
        log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[agent] Hypotheses log: {log_path}")

    def _log_final_code(self, iteration: int, code: str) -> None:
        logs_dir = self._experiment_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"iteration_{iteration}_final_code.py"
        log_path.write_text(code, encoding="utf-8")
        print(f"[agent] Final code log: {log_path}")

    def _write_preprocess(self, code: str) -> None:
        preprocess_path = _AGENT_DIR / "preprocess.py"
        preprocess_path.write_text(code + "\n", encoding="utf-8")
        print(f"[agent] preprocess.py updated ({len(code.splitlines())} lines).")

    @property
    def condition(self) -> str:
        if not self._use_analysis:
            return "agent_noinput"
        if self._use_history and self._use_contrastive:
            return "agent_contrastive"
        if self._use_history:
            return "agent_history"
        if self._use_contrastive:
            return "agent_contrastive_no_history"
        return "agent"

    def _model_folder(self) -> str:
        """Return a filesystem-safe folder name derived from the model string."""
        model = self._config.get("code_model", "unknown")
        # Strip provider prefix (e.g. "openai/gpt4o" → "gpt4o")
        return model.split("/")[-1].replace(".", "-")

    def _write_results(
        self,
        tracker: RunTracker,
        n_loops: int,
        n_docs: int,
        n_queries: int,
        baseline_results: dict,
        final_results: dict | None,
        baseline_val_results: dict | None = None,
        final_val_results: dict | None = None,
    ) -> None:
        results_dir = _PROJECT_ROOT / "results" / self._model_folder()
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"{self.condition}_{timestamp}.json"

        eval_m = final_results.get("metrics", {}) if final_results else {}
        val_m = final_val_results.get("metrics", {}) if final_val_results else {}
        payload = {
            "condition": self.condition,
            "model": self._config.get("code_model"),
            "loops": n_loops,
            "split": getattr(self, "split", "tip_of_the_tongue"),
            "seed": 42,
            "n_docs": n_docs,
            "n_queries": n_queries,
            # Held-out eval set (authoritative)
            "baseline_recall_100": baseline_results.get("recall_at_k"),
            "baseline_ndcg_10": baseline_results.get("ndcg"),
            "final_recall_100": eval_m.get("recall_at_100"),
            "final_ndcg_10": eval_m.get("ndcg_at_10"),
            "improvement_recall_100": (
                round(eval_m.get("recall_at_100", 0) - baseline_results.get("recall_at_k", 0), 4)
                if final_results else None
            ),
            # Validation set (agent's training signal)
            "baseline_val_recall_100": baseline_val_results.get("recall_at_k") if baseline_val_results else None,
            "baseline_val_ndcg_10": baseline_val_results.get("ndcg") if baseline_val_results else None,
            "final_val_recall_100": val_m.get("recall_at_100"),
            "final_val_ndcg_10": val_m.get("ndcg_at_10"),
            "latency": tracker.to_dict(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[agent] Results saved → {out_path}")

        # Also save a copy in the experiment directory
        if hasattr(self, '_experiment_dir') and self._experiment_dir:
            exp_results = self._experiment_dir / "results.json"
            exp_results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[agent] Results copy → {exp_results}")

    # --- Main loop ---

    def run(self, n_loops: int) -> None:
        """Override AgentRunner.run() with analysis+hypothesis loop."""

        # Load data
        corpus_size = self._config.get("corpus_size", None)
        documents, val_queries, eval_queries = _load_data(self.split, corpus_size=corpus_size)
        self._documents = documents
        self._val_queries = val_queries
        self._eval_queries = eval_queries
        print(f"\n{'='*60}")
        print(f"  Split       : {self.split}")
        print(f"  Documents   : {len(documents)}")
        print(f"  Val queries : {len(val_queries)}")
        print(f"  Eval queries: {len(eval_queries)}")
        print(f"{'='*60}\n")

        # Start BM25 server (must be up before baseline eval)
        self._ensure_server_running()

        # Compute baseline by running baseline preprocessor on the current corpus
        print(f"\n{'#'*60}")
        print(f"# Baseline (raw documents, no preprocessing) — computed on current corpus")
        print(f"{'#'*60}")
        val_baseline_results = self._compute_baseline(queries=val_queries)
        eval_baseline_results = self._compute_baseline(queries=eval_queries)
        print(f"  Eval Recall@100 : {eval_baseline_results['recall_at_k']:.4f}")
        print(f"  Eval nDCG@10    : {eval_baseline_results['ndcg']:.4f}")

        # Create per-experiment log directory
        model_name = self._config.get("code_model", "unknown_model").replace("/", "_")
        exp_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._experiment_dir = _PROJECT_ROOT / "ablation_experiments" / f"{model_name}_{self.condition}_{exp_timestamp}"
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        _setup_debug_logger(self._experiment_dir)
        print(f"[agent] Experiment logs → {self._experiment_dir}")
        logger.info(
            "Run start: model=%s condition=%s split=%s loops=%d baseline_recall@100=%.4f baseline_ndcg@10=%.4f",
            model_name, self.condition, self.split, n_loops,
            eval_baseline_results.get("recall_at_k", 0.0),
            eval_baseline_results.get("ndcg", 0.0),
        )

        # Create tracker + sub-agents + journal
        tracker = RunTracker()
        n_val = len(val_queries)
        n_eval = len(eval_queries)
        analysis_agent = AnalysisAgent(
            self._config, tracker=tracker, split=self.split,
            n_val_queries=n_val, n_eval_queries=n_eval,
        )
        code_agent = CodeAgent(
            self._config, tracker=tracker, split=self.split, log_dir=self._experiment_dir,
            n_val_queries=n_val, n_eval_queries=n_eval,
        )
        max_hypotheses = self._config.get("max_hypotheses", 4)
        all_past_hypotheses: list[dict] = []  # track across loops
        journal = RunJournal(self._experiment_dir)

        # Track globally best code + recall@100 (from harness eval) across all loops
        best_recall_100: float = eval_baseline_results.get("recall_at_k", 0.0)
        best_code: str = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8")

        for i in range(n_loops):
            print(f"\n{'#'*60}")
            print(f"# Loop {i + 1} / {n_loops}")
            print(f"{'#'*60}")
            logger.info("=== Loop %d/%d start ===", i + 1, n_loops)

            # Full harness eval — authoritative recall@100 for this loop's starting point
            try:
                raw_results = self.run_eval(iteration=i * 2, queries=eval_queries)
                eval_log = self._experiment_dir / f"iteration_{i}_eval.json"
                eval_log.write_text(json.dumps(raw_results, indent=2, default=str), encoding="utf-8")
            except Exception as e:
                logger.exception("Eval failed on loop %d", i + 1)
                print(f"[agent] Eval failed (loop {i + 1}): {e}")
                continue

            # Anchor: harness recall@100 at the start of this loop
            loop_start_recall_100 = raw_results["metrics"]["recall_at_100"]
            print(f"[agent] Loop {i+1} starting Eval recall@100: {loop_start_recall_100:.4f} "
                  f"(global best: {best_recall_100:.4f})")

            current_code = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8")

            # Rebuild "current" BM25 index on server
            print("[agent] Building 'current' index on BM25 server ...")
            try:
                chunks = self._preprocess_with_current_code(documents, current_code)
                self._client.build_index("current", chunks, persist=False)
                print(f"[agent] 'current' index built with {len(chunks)} chunks.")
            except Exception as e:
                logger.exception("Index build failed on loop %d", i + 1)
                print(f"[agent] Index build failed: {e}")
                continue

            # Enrich eval results with per-query data, record journal, and run analysis.
            # All three are skipped for agent_noinput: val per-query data only feeds analysis
            # + journal, and journal is only consulted when use_history is on (also off here).
            val_raw_results: dict | None = None
            if self._use_analysis:
                print("[agent] Enriching eval results with validation per-query data ...")
                try:
                    from .eval_utils import run_subset_eval
                    val_summary = run_subset_eval("current", val_queries, self._client, top_k=100)
                    val_raw_results = {
                        "metrics": {
                            "recall_at_10": val_summary.recall_at_10,
                            "recall_at_100": val_summary.recall_at_100,
                            "ndcg_at_10": val_summary.ndcg_at_10,
                        }
                    }
                    val_raw_results = self._enrich_eval_results(val_raw_results, val_queries, self._client)
                    print(f"[agent] Enriched with {len(val_raw_results.get('query_results', []))} val query results.")
                    val_log = self._experiment_dir / f"iteration_{i}_val.json"
                    val_log.write_text(json.dumps(val_raw_results, indent=2, default=str), encoding="utf-8")
                except Exception as e:
                    logger.exception("Val Enrichment failed on loop %d", i + 1)
                    print(f"[agent] Val Enrichment failed: {e}")
                    continue

                # Record iteration in journal with both val and eval metrics
                journal.record_iteration(
                    iteration=i,
                    eval_results=val_raw_results,
                    eval_results_harness=raw_results,
                )

                # Analysis agent uses validation results
                print("[agent] Running analysis agent on validation data...")
                try:
                    analysis_result = analysis_agent.analyze(
                        eval_results=val_raw_results,
                        baseline_results=val_baseline_results,
                        current_code=current_code,
                        client=self._client,
                        split=self.split,
                        journal_summary=journal.summary_for_prompt() if self._use_history else None,
                    )
                    self._log_analysis(i, analysis_result)
                    analysis_summary = analysis_result.summary
                except Exception as e:
                    logger.exception("Analysis agent failed on loop %d", i + 1)
                    print(f"[agent] Analysis failed: {e}")
                    continue
            else:
                analysis_summary = ""
                print("[agent] Skipping val enrichment, journal recording, and analysis (condition=agent_noinput).")

            # Hypothesis generation uses val queries.
            # NOTE: persistent_failure_ids is intentionally not passed any more —
            # priming the code agent with "MUST target these queries" caused
            # over-fitting on a small set of unfixable queries each iteration.
            print(f"[agent] Generating {max_hypotheses} hypotheses ...")
            query_lookup = {q.query_id: q.query_text for q in val_queries} if self._use_contrastive else None
            hypotheses = asyncio.run(code_agent.generate_hypotheses_async(
                analysis_summary,
                current_code,
                n=max_hypotheses,
                past_hypotheses=all_past_hypotheses if (all_past_hypotheses and self._use_history) else None,
                persistent_failure_ids=None,
                query_lookup=query_lookup,
            ))
            print(f"[agent] Generated {len(hypotheses)} hypotheses.")

            if not hypotheses:
                print("[agent] No hypotheses generated — skipping.")
                continue

            # Hypothesis testing — compare each against the BM25 server "current" index using validation queries
            print("[agent] Testing hypotheses on validation queries ...")
            hypothesis_results = []
            for h in hypotheses:
                print(f"[agent] Testing {h.id}: {h.description}")
                result = code_agent.test_hypothesis(
                    h, documents, val_queries, current_code, self._client
                )
                hypothesis_results.append(result)
            self._log_hypotheses(i, hypothesis_results)

            # Track all tested hypotheses for future loops + journal
            for r in hypothesis_results:
                all_past_hypotheses.append({
                    "id": r.hypothesis.id,
                    "description": r.hypothesis.description,
                    "delta_recall_100": r.delta_recall_100,
                    "delta_recall_10": r.delta_recall_10,
                    "delta_ndcg_10": r.delta_ndcg_10,
                    "proven": r.proven,
                    "notes": r.notes,
                    "improved_query_ids": r.improved_query_ids,
                    "regressed_query_ids": r.regressed_query_ids,
                })

            # Pick the single best hypothesis by recall@100 on the BM25 server (val queries).
            valid = [r for r in hypothesis_results if not r.error]
            if not valid:
                print("[agent] All hypotheses errored — preprocess.py unchanged.")
                continue

            best_hyp = max(valid, key=lambda r: r.hypothesis_recall_100)
            print(f"[agent] Best hypothesis: {best_hyp.hypothesis.id} "
                  f"val_recall@100={best_hyp.hypothesis_recall_100:.4f} "
                  f"(Δ{best_hyp.delta_recall_100:+.4f} vs val current, "
                  f"val baseline={best_hyp.baseline_recall_100:.4f})")

            # Record all hypothesis results in journal (mark adopted=False first)
            for r in hypothesis_results:
                adopted = (r is best_hyp and best_hyp.delta_recall_100 > 0)
                journal.record_hypothesis(
                    iteration=i,
                    h_id=r.hypothesis.id,
                    description=r.hypothesis.description,
                    rationale=r.hypothesis.rationale,
                    targeted_query_ids=r.hypothesis.query_ids_to_test,
                    delta_recall_100=r.delta_recall_100,
                    delta_recall_10=r.delta_recall_10,
                    delta_ndcg_10=r.delta_ndcg_10,
                    proven=r.proven,
                    adopted=adopted,
                    improved_query_ids=r.improved_query_ids,
                    regressed_query_ids=r.regressed_query_ids,
                    error=r.error,
                )

            proven_results = [r for r in valid if r.proven]
            candidate_eval_recall_100 = best_recall_100  # Default to best if not adopted

            if best_hyp.delta_recall_100 > 0:
                journal.set_iteration_adoption(i, best_hyp.hypothesis.id)

                if best_hyp.regressed_query_ids:
                    print(
                        f"[agent] ⚠ Overfitting: regresses {len(best_hyp.regressed_query_ids)} val queries "
                        f"({', '.join(best_hyp.regressed_query_ids[:5])}{'...' if len(best_hyp.regressed_query_ids) > 5 else ''})"
                    )

                # If multiple hypotheses proved, try synthesis instead of just picking best
                was_synthesized = False
                synthesized_from_ids: list[str] = []
                if len(proven_results) > 1:
                    print(f"[agent] {len(proven_results)} hypotheses proved — attempting synthesis ...")
                    synthesized = code_agent.generate_final_code(
                        analysis_summary, proven_results, current_code
                    )
                    if synthesized:
                        val_err = code_agent._validate_code(synthesized, documents)
                        if val_err:
                            logger.error("Synthesis validation failed on loop %d:\n%s", i, val_err)
                            print(f"[agent] Synthesis validation failed: {val_err[:80]} — falling back to best hypothesis.")
                            final_code = best_hyp.hypothesis.code
                        else:
                            # Test synthesized code on val queries — only adopt if it beats best individual hypothesis
                            try:
                                from .eval_utils import run_subset_eval
                                synth_chunks = self._preprocess_with_current_code(documents, synthesized)
                                self._client.build_index("synthesized", synth_chunks, persist=False)
                                synth_summary = run_subset_eval("synthesized", val_queries, self._client, top_k=100)
                                synth_recall = synth_summary.recall_at_100
                                print(f"[agent] Synthesized val recall@100={synth_recall:.4f} vs best hypothesis {best_hyp.hypothesis_recall_100:.4f}")
                                if synth_recall > best_hyp.hypothesis_recall_100:
                                    print(f"[agent] Synthesis beats best hypothesis — adopting.")
                                    final_code = synthesized
                                    was_synthesized = True
                                    synthesized_from_ids = [r.hypothesis.id for r in proven_results]
                                else:
                                    print(f"[agent] Synthesis did not beat best hypothesis — falling back to {best_hyp.hypothesis.id}.")
                                    final_code = best_hyp.hypothesis.code
                            except Exception as e:
                                logger.exception("Synthesis val eval failed on loop %d", i)
                                print(f"[agent] Synthesis val eval failed: {e} — falling back to best hypothesis.")
                                final_code = best_hyp.hypothesis.code
                    else:
                        print(f"[agent] Synthesis failed — falling back to best hypothesis.")
                        final_code = best_hyp.hypothesis.code
                else:
                    print(f"[agent] Adopting {best_hyp.hypothesis.id} directly.")
                    final_code = best_hyp.hypothesis.code

                self._log_final_code(i, final_code)

                # Now evaluate the chosen code on the evaluation queries
                self._write_preprocess(final_code)
                print("[agent] Running authoritative harness eval on adopted code ...")
                try:
                    candidate_results = self.run_eval(iteration=i * 2 + 1, queries=eval_queries)
                    candidate_eval_recall_100 = candidate_results["metrics"]["recall_at_100"]
                    print(f"[agent] Adopted Eval recall@100={candidate_eval_recall_100:.4f} "
                          f"(global best so far: {best_recall_100:.4f})")
                except Exception as e:
                    logger.exception("Harness eval of adopted code failed on loop %d", i)
                    print(f"[agent] Harness eval of adopted code failed: {e} — reverting to pre-loop code.")
                    self._write_preprocess(current_code)
                    continue

                # Write accepted hypothesis JSON
                accepted_data = {
                    "iteration": i,
                    "adopted_hypothesis": {
                        "id": best_hyp.hypothesis.id,
                        "description": best_hyp.hypothesis.description,
                        "rationale": best_hyp.hypothesis.rationale,
                        "delta_val_recall_100": best_hyp.delta_recall_100,
                        "delta_val_recall_10": best_hyp.delta_recall_10,
                        "delta_val_ndcg_10": best_hyp.delta_ndcg_10,
                    },
                    "synthesized": was_synthesized,
                    "synthesized_from": synthesized_from_ids,
                    "proven_hypotheses": [
                        {"id": r.hypothesis.id, "description": r.hypothesis.description}
                        for r in proven_results
                    ],
                    "candidate_eval_recall_100": candidate_eval_recall_100,
                    "global_best_recall_100_before": best_recall_100,
                }
                accepted_path = self._experiment_dir / f"iteration_{i}_accepted.json"
                accepted_path.write_text(json.dumps(accepted_data, indent=2), encoding="utf-8")

                if candidate_eval_recall_100 > best_recall_100:
                    best_recall_100 = candidate_eval_recall_100
                    best_code = final_code
                    print(f"[agent] Global best updated → eval recall@100={best_recall_100:.4f}")
                else:
                    # Depending on policy, we might still keep it or revert. Currently, code keeps it written since we already wrote it.
                    # Wait, should we revert if it performs worse on eval? The agent only learns from val.
                    # Actually, if we adopt it, we keep it because that's our best guess. If eval goes down, it's overfitting, but we shouldn't peek at eval to decide whether to keep it! That would be a data leak.
                    print(f"[agent] Candidate adopted, but eval recall did not beat global best.")
                
                logger.info(
                    "Loop %d end: adopted=%s synthesized=%s candidate_eval_recall@100=%.4f best=%.4f",
                    i + 1, best_hyp.hypothesis.id, was_synthesized,
                    candidate_eval_recall_100, best_recall_100,
                )
            else:
                # No hypothesis improved — write accepted JSON indicating no adoption
                accepted_data = {
                    "iteration": i,
                    "adopted_hypothesis": None,
                    "reason": "no improvement",
                }
                accepted_path = self._experiment_dir / f"iteration_{i}_accepted.json"
                accepted_path.write_text(json.dumps(accepted_data, indent=2), encoding="utf-8")
                print(f"[agent] No hypothesis improved val recall over current — preprocess.py unchanged.")
                logger.info("Loop %d end: no adoption (best=%.4f)", i + 1, best_recall_100)

        # Final eval
        print(f"\n{'#'*60}")
        print(f"# Final eval (after {n_loops} loop{'s' if n_loops != 1 else ''})")
        print(f"{'#'*60}")
        final_results = None
        final_val_results = None
        try:
            final_results = self.run_eval(queries=eval_queries)
            final_recall = final_results["metrics"]["recall_at_100"]
            baseline_recall = eval_baseline_results.get("recall_at_k", 0.0)
            print(f"\n[agent] Improvement: Eval recall@100 {baseline_recall:.4f} → {final_recall:.4f} "
                  f"({final_recall - baseline_recall:+.4f})")
        except Exception as e:
            logger.exception("Final eval failed")
            print(f"[agent] Final eval failed: {e}")

        try:
            final_val_results = self.run_eval(queries=val_queries)
            final_val_recall = final_val_results["metrics"]["recall_at_100"]
            baseline_val_recall = val_baseline_results.get("recall_at_k", 0.0)
            print(f"[agent] Improvement: Val  recall@100 {baseline_val_recall:.4f} → {final_val_recall:.4f} "
                  f"({final_val_recall - baseline_val_recall:+.4f})")
        except Exception as e:
            logger.exception("Final val eval failed")
            print(f"[agent] Final val eval failed: {e}")

        # Write results JSON
        self._write_results(
            tracker=tracker,
            n_loops=n_loops,
            n_docs=len(documents),
            n_queries=len(eval_queries),
            baseline_results=eval_baseline_results,
            final_results=final_results,
            baseline_val_results=val_baseline_results,
            final_val_results=final_val_results,
        )

        # Clean up server
        self._kill_server()
