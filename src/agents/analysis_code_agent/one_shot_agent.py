"""
one_shot_agent.py — Single-call LLM baseline.

Makes one LLM call with the task description + baseline eval feedback,
writes the returned preprocess.py, runs eval, and saves results.
No iterative loop, no bash investigation, no hypothesis history.
"""
from __future__ import annotations

import json
import logging
import os
import re
import pathlib
import time
import datetime

import yaml
from dotenv import load_dotenv
from .llm_call import completion

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent

logger = logging.getLogger("analysis_code_agent")


def run_one_shot(split: str = "tip_of_the_tongue", model: str | None = None, api_base: str | None = None, max_distractors: int = 9000) -> None:
    """Run the one-shot baseline: one LLM call → eval → save results."""

    config_path = _AGENT_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model = model or config.get("code_model", "openai/gpt4o")
    # Only pass api_key explicitly when using the proxy (api_base set).
    # For native provider endpoints (e.g. Gemini), let LiteLLM read the key from env.
    proxy_api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LITE_LLM_KEY")
        or os.environ.get("LITELLM_API_KEY")
        or ""
    )
    # If api_base explicitly provided, use it. If model was overridden but no api_base given,
    # use None so LiteLLM routes to the provider's native endpoint (e.g. Google for gemini/).
    # Only fall back to config api_base when using the default model (no override).
    if api_base is not None:
        pass  # use as-is
    elif model != config.get("code_model"):
        api_base = None  # model was overridden — use native endpoint
    else:
        api_base = config.get("api_base")
    temperature = config.get("code_temperature", 1.0)
    if max_distractors == 9000:  # default value, check config
        max_distractors = config.get("max_distractors", 9000)

    # --- Load data ---
    import subprocess, atexit, sys
    from .agent import _load_data
    from .run_tracker import RunTracker
    from .eval_utils import load_preprocessor_from_code, run_subset_eval
    from .bm25_client import BM25Client

    _EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"
    if str(_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(_EVAL_DIR))

    # --- Create experiment dir + debug log up front so errors are captured ---
    from .agent import _setup_debug_logger
    model_short_early = model.replace("/", "_")
    exp_timestamp_early = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir_early = _PROJECT_ROOT / "ablation_experiments" / f"{model_short_early}_one_shot_{exp_timestamp_early}"
    experiment_dir_early.mkdir(parents=True, exist_ok=True)
    _setup_debug_logger(experiment_dir_early)
    logger.info("one_shot run start: model=%s split=%s", model, split)

    print("[one_shot] Loading data ...")
    documents, queries = _load_data(split)
    print(f"[one_shot] {len(documents)} docs, {len(queries)} queries.")

    # --- Start BM25 server ---
    server_port = config.get("server_port", 8765)
    persist_dir = str(_AGENT_DIR / ".bm25_cache")
    server_path = _AGENT_DIR / "bm25_server.py"
    server_proc = subprocess.Popen(
        ["uv", "run", "python", str(server_path), "--port", str(server_port), "--persist-dir", persist_dir],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(lambda: server_proc.terminate())
    time.sleep(3)
    client = BM25Client(base_url=f"http://localhost:{server_port}")

    # --- Evaluate baseline preprocessor on the current corpus ---
    baseline_preprocess_path = _AGENT_DIR.parent / "baseline" / "preprocess.py"
    baseline_code = baseline_preprocess_path.read_text(encoding="utf-8")
    baseline_recall = 0.0
    baseline_ndcg = 0.0

    try:
        baseline_preprocessor = load_preprocessor_from_code(baseline_code)
        baseline_chunks = baseline_preprocessor.preprocess(documents)
        client.build_index("one_shot_baseline", baseline_chunks, persist=False)
        baseline_eval = run_subset_eval("one_shot_baseline", queries, client, top_k=100)
        baseline_recall = baseline_eval.recall_at_100
        baseline_ndcg = baseline_eval.ndcg_at_10
        miss_ids = [r.query_id for r in baseline_eval.per_query if not r.hit_at_100][:30]
    except Exception as e:
        logger.exception("one_shot baseline eval failed")
        print(f"[one_shot] Baseline eval failed: {e} — using empty miss list.")
        miss_ids = []

    # --- Build prompt ---
    from .analysis_agent import load_corpus_description
    system_prompt = (_AGENT_DIR / "context" / "CODE_SYSTEM.md").read_text(encoding="utf-8")
    context_info = (
        (_AGENT_DIR.parent / "CONTEXT.md").read_text(encoding="utf-8")
        + "\n"
        + load_corpus_description(split)
    )
    current_code = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8")

    miss_section = ""
    if miss_ids:
        miss_section = (
            f"\n## Currently Failing Queries (not retrieved in top-100)\n"
            f"These {len(miss_ids)} query IDs fail with the baseline preprocessor: "
            f"{', '.join(miss_ids)}\n"
            f"Design your preprocessing to improve recall for these queries.\n"
        )

    prompt = f"""## Task
You are given a BM25 retrieval system over a document corpus.
The current preprocessing achieves:
  - Recall@100: {baseline_recall:.4f}
  - nDCG@10:    {baseline_ndcg:.4f}

Write a single, complete preprocess.py that will improve these scores.
You have ONE attempt — make it count.

## Dataset Info
{context_info}

## Current preprocess.py (starting point)
```python
{current_code}
```
{miss_section}
Output a single complete Python file inside a ```python ... ``` block.
The file must define `class Preprocessor(BasePreprocessor)` with a `preprocess(self, docs) -> List[Chunk]` method.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # --- Call LLM ---
    tracker = RunTracker()
    print(f"[one_shot] Calling {model} ...")
    t0 = time.time()
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            api_key=proxy_api_key if api_base else None,
            api_base=api_base,
            timeout=config.get("code_llm_timeout"),
        )
        tracker.record_llm_call(response, time.time() - t0, agent="one_shot")
        text = response.choices[0].message.content or ""
    except Exception as e:
        logger.exception("one_shot LLM call failed (model=%s)", model)
        print(f"[one_shot] LLM call failed: {e}")
        server_proc.terminate()
        return

    # --- Extract and write code ---
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if not match:
        logger.warning("one_shot: No python block in response. Raw text:\n%s", text)
        print("[one_shot] No python block found in response.")
        server_proc.terminate()
        return

    code = match.group(1).strip()
    preprocess_path = _AGENT_DIR / "preprocess.py"
    preprocess_path.write_text(code + "\n", encoding="utf-8")
    print(f"[one_shot] preprocess.py written ({len(code.splitlines())} lines).")

    # --- Eval ---
    print("[one_shot] Running eval ...")
    final_results = None
    try:
        preprocessor = load_preprocessor_from_code(code)
        chunks = preprocessor.preprocess(documents)
        client.build_index("one_shot_eval", chunks, persist=False)
        summary = run_subset_eval("one_shot_eval", queries, client, top_k=100)
        final_results = {
            "recall_at_100": summary.recall_at_100,
            "recall_at_10": summary.recall_at_10,
            "ndcg_at_10": summary.ndcg_at_10,
        }
        print(
            f"\n  Recall@10  : {summary.recall_at_10:.4f}\n"
            f"  Recall@100 : {summary.recall_at_100:.4f}\n"
            f"  nDCG@10    : {summary.ndcg_at_10:.4f}\n"
            f"\n[one_shot] Improvement: recall@100 {baseline_recall:.4f} → {summary.recall_at_100:.4f} "
            f"({summary.recall_at_100 - baseline_recall:+.4f})"
        )
    except Exception as e:
        logger.exception("one_shot eval failed")
        print(f"[one_shot] Eval failed: {e}")

    server_proc.terminate()

    # --- Save results ---
    model_folder = model.split("/")[-1].replace(".", "-")
    results_dir = _PROJECT_ROOT / "results" / model_folder
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"one_shot_{timestamp}.json"
    payload = {
        "condition": "one_shot",
        "loops": 1,
        "split": split,
        "model": model,
        "seed": 42,
        "n_docs": len(documents),
        "n_queries": len(queries),
        "baseline_recall_100": baseline_recall,
        "baseline_ndcg_10": baseline_ndcg,
        "final_recall_100": final_results.get("recall_at_100") if final_results else None,
        "final_ndcg_10": final_results.get("ndcg_at_10") if final_results else None,
        "improvement_recall_100": (
            round(final_results["recall_at_100"] - baseline_recall, 4) if final_results else None
        ),
        "latency": tracker.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[one_shot] Results saved → {out_path}")

    # --- Save to experiment directory (created early so debug.log is already there) ---
    experiment_dir = experiment_dir_early
    (experiment_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (experiment_dir / "preprocess.py").write_text(code + "\n", encoding="utf-8")
    print(f"[one_shot] Experiment logs → {experiment_dir}")
