"""
test_preprocessing_split.py – Evaluation harness with iteration tracking for LLM agents.

Tracks metrics across iterations when called from main.py agent runner.
"""

from __future__ import annotations

import sys
import pathlib
import json
import argparse
import importlib
import subprocess
import re
import math
from typing import List, Dict, Optional
from datetime import datetime

# [Keep all existing imports and setup code]
_EVAL_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_EVAL_DIR / "scripts"))

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from schema import Document, EvalQuery
from base import BasePreprocessor
from build_index import BM25Index

DATA_DIR = _PROJECT_ROOT / "data"
RESULTS_DIR = _PROJECT_ROOT / "results"

try:
    from crumb_eval import evaluate as crumb_evaluate
    CRUMB_EVAL_AVAILABLE = True
except ImportError:
    CRUMB_EVAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# [Keep all existing data loading functions unchanged]
# ---------------------------------------------------------------------------

def _load_documents(split: str = None) -> tuple[List[Document], str]:
    """Load documents from split-specific cache, or fallback to default."""
    if split:
        docs_file = DATA_DIR / split / "documents.jsonl"
        if docs_file.exists():
            print(f"✓ Using cached data for split: {split}")
            docs = []
            with docs_file.open(encoding="utf-8") as f:
                for line in f:
                    docs.append(Document(**json.loads(line)))
            return docs, split
    # [rest of function unchanged]
    raise FileNotFoundError(f"No data found!")


def _load_queries(split: str = None) -> tuple[List[EvalQuery], str]:
    """Load queries from split-specific cache, or fallback to default."""
    if split:
        queries_file = DATA_DIR / split / "evaluation_queries.jsonl"
        if queries_file.exists():
            queries = []
            with queries_file.open(encoding="utf-8") as f:
                for line in f:
                    q = json.loads(line)
                    queries.append(EvalQuery(
                        query_id=q['query_id'],
                        query_text=q['query_content'] if 'query_content' in q else q.get('query_text', ''),
                        relevant_doc_ids=q['relevant_doc_ids']
                    ))
            return queries, split
        # Fallback to queries.jsonl for older datasets
        queries_file = DATA_DIR / split / "queries.jsonl"
        if queries_file.exists():
            queries = []
            with queries_file.open(encoding="utf-8") as f:
                for line in f:
                    q = json.loads(line)
                    queries.append(EvalQuery(
                        query_id=q['query_id'],
                        query_text=q['query_content'] if 'query_content' in q else q.get('query_text', ''),
                        relevant_doc_ids=q['relevant_doc_ids']
                    ))
            return queries, split

    default_queries = DATA_DIR / "evaluation_queries.jsonl"
    if default_queries.exists():
        queries = []
        with default_queries.open(encoding="utf-8") as f:
            for line in f:
                queries.append(EvalQuery(**json.loads(line)))
        return queries, "default"

    default_queries = DATA_DIR / "queries.jsonl"
    if default_queries.exists():
        queries = []
        with default_queries.open(encoding="utf-8") as f:
            for line in f:
                queries.append(EvalQuery(**json.loads(line)))
        return queries, "default"

    raise FileNotFoundError(f"No queries found!")


def _list_available_splits() -> List[str]:
    """List all cached splits."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and ((d / "evaluation_queries.jsonl").exists() or (d / "queries.jsonl").exists())
    )

# ---------------------------------------------------------------------------
# Results saving WITH iteration tracking
# ---------------------------------------------------------------------------

def parse_crumb_output(output_text: str) -> Dict[str, float]:
    """Parse CRUMB eval text output into a metrics dict."""
    metrics = {}
    
    # More flexible regex that handles various formats
    # Matches lines like: "nDCG@10: 0.4909" or "P@5: 0.1806"
    pattern = r'([A-Za-z@0-9]+)\s*:\s*([\d.]+)'
    
    for line in output_text.split('\n'):
        line = line.strip()
        match = re.search(pattern, line)
        if match:
            metric_name, value = match.groups()
            try:
                metrics[metric_name] = float(value)
            except ValueError:
                continue
    
    return metrics


def _save_results(
    results: dict,
    split: str,
    agent: str,
    n_docs: int,
    top_k: int,
    iteration: Optional[int] = None  # NEW: iteration parameter
) -> pathlib.Path:
    """Save results with optional iteration number in filename."""
    results_dir = RESULTS_DIR / split
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Add iteration to filename if provided
    if iteration is not None:
        filename = f"{agent}_{n_docs}docs_{top_k}_iter{iteration}.json"
    else:
        filename = f"{agent}_{n_docs}docs_{top_k}.json"
    
    path = results_dir / filename
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {path}")
    return path


def _save_query_results(
    query_results: List[Dict],
    split: str,
    agent: str,
    n_docs: int,
    top_k: int,
    iteration: Optional[int] = None  # NEW
) -> pathlib.Path:
    """Save query results with optional iteration number."""
    results_dir = RESULTS_DIR / split
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if iteration is not None:
        filename = f"{agent}_{n_docs}docs_{top_k}_iter{iteration}_results.json"
    else:
        filename = f"{agent}_{n_docs}docs_{top_k}_results.json"
    
    path = results_dir / filename
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(query_results, f, indent=2)
    
    print(f"✓ Query results saved to: {path}")
    return path


def _save_crumb_format(
    query_results: List[Dict],
    split: str,
    agent: str,
    n_docs: int,
    top_k: int,
    iteration: Optional[int] = None  # NEW
) -> pathlib.Path:
    """Save CRUMB format with optional iteration number."""
    results_dir = RESULTS_DIR / split
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if iteration is not None:
        filename = f"{agent}_{n_docs}docs_{top_k}_iter{iteration}_crumb.jsonl"
    else:
        filename = f"{agent}_{n_docs}docs_{top_k}_crumb.jsonl"
    
    path = results_dir / filename
    
    with path.open('w', encoding='utf-8') as f:
        for qr in query_results:
            entry = {
                "query": {"id": qr["query_id"]},
                "items": [
                    {"id": str(doc_id), "score": float(score)}
                    for doc_id, score in qr["ranked_docs"]
                ]
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"✓ CRUMB format saved to: {path}")
    return path


def _save_iteration_summary(
    all_iterations: List[dict],
    split: str,
    agent: str
) -> pathlib.Path:
    """
    Save summary of all iterations showing improvement over time.
    
    Creates: results/<split>/<agent>_iterations.json
    """
    results_dir = RESULTS_DIR / split
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{agent}_iterations.json"
    path = results_dir / filename
    
    summary = {
        "agent": agent,
        "split": split,
        "total_iterations": len(all_iterations),
        "last_updated": datetime.now().isoformat(),
        "iterations": all_iterations
    }
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Iteration summary saved to: {path}")
    
    # Print improvement summary
    if len(all_iterations) > 1:
        print(f"\n{'='*60}")
        print("IMPROVEMENT OVER ITERATIONS")
        print(f"{'='*60}")
        print(f"{'Iter':<6} {'Recall@100':<12} {'Recall@1000':<12} {'nDCG@10':<12} {'Chunks/Doc':<12}")
        print("-"*60)
        for it in all_iterations:
            metrics = it.get('metrics', {})
            config = it.get('config', {})
            iter_num = it.get('iteration', '?')
            print(
                f"{iter_num:<6} "
                f"{metrics.get('recall_at_100', 0):<12.4f} "
                f"{metrics.get('recall_at_1000', 0):<12.4f} "
                f"{metrics.get('ndcg_at_10', 0):<12.4f} "
                f"{config.get('chunks_per_doc', 0):<12.2f}"
            )
        
        # Show delta from first to last
        first = all_iterations[0]['metrics']
        last = all_iterations[-1]['metrics']
        print("-"*60)
        print(f"{'Δ':<6} "
              f"{last.get('recall_at_100', 0) - first.get('recall_at_100', 0):+.4f}       "
              f"{last.get('recall_at_1000', 0) - first.get('recall_at_1000', 0):+.4f}       "
              f"{last.get('ndcg_at_10', 0) - first.get('ndcg_at_10', 0):+.4f}")
        print(f"{'='*60}\n")
    
    return path


def _load_iteration_history(split: str, agent: str) -> List[dict]:
    """Load existing iteration history if it exists."""
    results_dir = RESULTS_DIR / split
    history_file = results_dir / f"{agent}_iterations.json"
    
    if not history_file.exists():
        return []
    
    try:
        with history_file.open('r') as f:
            data = json.load(f)
            return data.get('iterations', [])
    except Exception:
        return []


def _load_all_results(split: str) -> List[dict]:
    """Load all saved results for comparison."""
    results_dir = RESULTS_DIR / split
    if not results_dir.exists():
        return []
    
    results = []
    for f in results_dir.glob("*.json"):
        # Skip crumb format files, query results files, and iteration summaries
        if "_crumb" not in f.name and "_results" not in f.name and "_iterations" not in f.name:
            try:
                with f.open('r', encoding='utf-8') as fh:
                    results.append(json.load(fh))
            except Exception as e:
                print(f"Warning: Could not load {f.name}: {e}")
    return results


# ---------------------------------------------------------------------------
# Evaluation WITH iteration tracking
# ---------------------------------------------------------------------------

def evaluate(
    preprocessor: BasePreprocessor,
    split: str,
    top_k: int = 100,
    save_results: bool = True,
    iteration: Optional[int] = None,  # NEW: iteration number
    track_iterations: bool = False,   # NEW: enable iteration tracking
) -> dict:
    """
    Run evaluation with optional iteration tracking.
    
    Args:
        iteration: If provided, saves with iteration number in filename
        track_iterations: If True, appends to iteration history file
    """
    docs, actual_split = _load_documents(split)
    queries, _ = _load_queries(split)

    agent_name = preprocessor.name or type(preprocessor).__name__
    
    # Print header with iteration info if applicable
    iter_str = f" (Iteration {iteration})" if iteration is not None else ""
    print(f"\n{'='*60}")
    print(f"Agent       : {agent_name}{iter_str}")
    print(f"Split       : {actual_split}")
    print(f"Description : {preprocessor.description or '(none)'}")
    print(f"{'='*60}")
    print(f"Preprocessing {len(docs)} documents ...")

    # Preprocess documents
    chunks = preprocessor.preprocess(docs)
    
    # Normalize chunk doc_ids
    for c in chunks:
        c.doc_id = str(c.doc_id)
    
    print(f"  -> {len(chunks)} chunks  ({len(chunks)/len(docs):.2f} avg per doc)")

    print("Building BM25 index ...")
    index = BM25Index(chunks, candidate_k=10000, agg="max")

    # [Keep all existing retrieval and metrics computation code]
    query_results = []
    recall_at_100_hits = 0
    recall_at_1000_hits = 0
    ndcg_at_10_total = 0.0
    
    for query in queries:
        chunk_results = index.search(query.query_text, top_k=top_k * 10)
        
        doc_scores: Dict[str, float] = {}
        for chunk, score in chunk_results:
            doc_id = str(chunk.doc_id)
            sc = float(score)
            if doc_id not in doc_scores or sc > doc_scores[doc_id]:
                doc_scores[doc_id] = sc
        
        # Sort ALL docs, don't truncate yet (we need 1000 for recall calculation)
        ranked_docs_full = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))
        ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs_full]
        
        # For saving results, limit to top_k
        ranked_docs = ranked_docs_full[:top_k]
        
        query_results.append({
            "query_id": query.query_id,
            "ranked_docs": ranked_docs,
        })
        
        relevant = set(map(str, query.relevant_doc_ids))
        n_relevant = len(relevant) or 1

        recall_at_100_hits += len(relevant & set(ranked_doc_ids[:100])) / n_relevant
        recall_at_1000_hits += len(relevant & set(ranked_doc_ids[:1000])) / n_relevant
        
        dcg = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids[:10], start=1):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), 10) + 1))
        ndcg_at_10_total += (dcg / idcg) if idcg > 0 else 0.0
    
    n_queries = len(queries)
    recall_at_100 = recall_at_100_hits / n_queries
    recall_at_1000 = recall_at_1000_hits / n_queries
    ndcg_at_10 = ndcg_at_10_total / n_queries
    
    print(f"\nQuick metrics ({n_queries} queries, top-{top_k}):")
    print(f"  Recall@100  : {recall_at_100:.4f}")
    print(f"  Recall@1000 : {recall_at_1000:.4f}")
    print(f"  nDCG@10    : {ndcg_at_10:.4f}")

    # Build results structure
    results = {
        "agent": agent_name,
        "split": actual_split,
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,  # NEW: track iteration number
        "config": {
            "top_k": top_k,
            "n_docs": len(docs),
            "n_queries": len(queries),
            "n_chunks": len(chunks),
            "chunks_per_doc": len(chunks) / len(docs),
        },
        "metrics": {
            "recall_at_100": recall_at_100,
            "recall_at_1000": recall_at_1000,
            "ndcg_at_10": ndcg_at_10,
        },
        "crumb_metrics": None,
    }

    if save_results:
        # Save with iteration number if provided
        _save_results(results, actual_split, agent_name, len(docs), top_k, iteration)
        _save_query_results(query_results, actual_split, agent_name, len(docs), top_k, iteration)
        crumb_path = _save_crumb_format(query_results, actual_split, agent_name, len(docs), top_k, iteration)
        
        # [Keep existing CRUMB eval code...]
        if CRUMB_EVAL_AVAILABLE:
            print("\n" + "="*60)
            print("RUNNING CRUMB EVAL (Official Metrics)")
            print("="*60)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "crumb_eval.evaluate",
                     "--run_path", str(crumb_path),
                     "--task_name", actual_split,
                     "--max_p", "auto"],
                    capture_output=True, text=True, timeout=300,
                )
                crumb_output = result.stdout
                print(crumb_output)
                crumb_metrics = parse_crumb_output(crumb_output)
                if crumb_metrics:
                    results["crumb_metrics"] = crumb_metrics
                    print(f"\n✓ Captured {len(crumb_metrics)} metrics from CRUMB eval")
            except Exception as e:
                print(f"\n⚠️  CRUMB eval error: {e}")
            
            _save_results(results, actual_split, agent_name, len(docs), top_k, iteration)
            print("="*60)
        
        # NEW: Track iterations if enabled
        if track_iterations and iteration is not None:
            history = _load_iteration_history(actual_split, agent_name)
            history.append(results)
            _save_iteration_summary(history, actual_split, agent_name)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test preprocessing agent on CRUMB split with iteration tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent name (folder under src/agents/)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="CRUMB split name (if not provided, lists available splits)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top results to retrieve (default: 100)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results (just print metrics)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration number (for tracking agent improvement over time)",
    )
    parser.add_argument(
        "--track-iterations",
        action="store_true",
        help="Enable iteration tracking and summary generation",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with previous results after evaluation",
    )

    args = parser.parse_args()

    # List available splits if no split provided
    if not args.split:
        available = _list_available_splits()
        if available:
            print("Available splits:")
            for s in available:
                print(f"  - {s}")
        else:
            print("No cached splits found. Run get_data.py first.")
        return

    # Load the agent's preprocessor
    agent_path = _PROJECT_ROOT / "src" / "agents" / args.agent
    if not agent_path.exists():
        print(f"❌ Agent not found: {args.agent}")
        print(f"   Expected: {agent_path}")
        return

    # Import the preprocessor
    try:
        spec = importlib.util.spec_from_file_location(
            "preprocess_module",
            agent_path / "preprocess.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preprocessor = module.Preprocessor()
    except Exception as e:
        print(f"❌ Failed to load agent '{args.agent}': {e}")
        return

    # Run evaluation
    results = evaluate(
        preprocessor=preprocessor,
        split=args.split,
        top_k=args.top_k,
        save_results=not args.no_save,
        iteration=args.iteration,
        track_iterations=args.track_iterations,
    )

    # Compare with previous results if requested
    if args.compare and not args.no_save:
        print(f"\n{'='*60}")
        print("COMPARISON WITH PREVIOUS RUNS")
        print(f"{'='*60}")
        
        all_results = _load_all_results(args.split)
        if len(all_results) > 1:
            # Sort by timestamp
            all_results.sort(key=lambda x: x.get('timestamp', ''))
            
            print(f"\n{'Agent':<25} {'Recall@100':<12} {'Recall@1000':<12} {'nDCG@10':<12} {'Timestamp':<20}")
            print("-"*85)
            
            for r in all_results:
                agent = r.get('agent', 'unknown')
                metrics = r.get('metrics', {})
                timestamp = r.get('timestamp', '')[:19]  # Truncate to datetime
                iter_num = r.get('iteration')
                
                agent_display = f"{agent} (iter {iter_num})" if iter_num is not None else agent
                
                print(
                    f"{agent_display:<25} "
                    f"{metrics.get('recall_at_100', 0):<12.4f} "
                    f"{metrics.get('recall_at_1000', 0):<12.4f} "
                    f"{metrics.get('ndcg_at_10', 0):<12.4f} "
                    f"{timestamp:<20}"
                )
        else:
            print("No previous results to compare.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()