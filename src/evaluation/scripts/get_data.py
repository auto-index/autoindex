"""
get_data.py — Downloads corpus + queries for CRUMB splits using STREAMING.

Usage:
    python -m src.evaluation.scripts.get_data --split paper_retrieval
    python -m src.evaluation.scripts.get_data --split tip_of_the_tongue
    python -m src.evaluation.scripts.get_data --split code_retrieval --limit 5000

Caches to: data/<split>/documents.jsonl and data/<split>/queries.jsonl
Re-running with --limit overwrites existing cached data for that split.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm

SPLITS_JSON = Path(__file__).parents[3] / "query_splits.json"


def _apply_split(all_queries: List[Dict], split: str) -> tuple[List[Dict], List[Dict]]:
    """Partition queries using query_splits.json; falls back to 1:5 positional split."""
    if SPLITS_JSON.exists():
        config = json.loads(SPLITS_JSON.read_text())
        if split in config:
            by_id = {q["query_id"]: q for q in all_queries}
            val = [by_id[qid] for qid in config[split]["validation"] if qid in by_id]
            evl = [by_id[qid] for qid in config[split]["evaluation"] if qid in by_id]
            return val, evl
    n_val = max(1, len(all_queries) // 6)
    return all_queries[:n_val], all_queries[n_val:]

SPLIT_MAP = {
    "tip_of_the_tongue":              "tip_of_the_tongue",
    "paper_retrieval":                "paper_retrieval",
    "stack_exchange":                 "stack_exchange",
    "clinical_trial":                 "clinical_trial",
    "legal_qa":                       "legal_qa",
    "theorem_retrieval":              "theorem_retrieval",
    "code_retrieval":                 "code_retrieval",
    "set_operation_entity_retrieval": "set_operation_entity_retrieval",
}


def get_cache_dir(split: str) -> Path:
    """Get cache directory for a split: data/<split>/"""
    base_dir = Path(__file__).parents[3] / "data"
    cache_dir = base_dir / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_queries(split: str, n_queries: int = None) -> tuple[List[Dict], List[Dict]]:
    crumb_name = SPLIT_MAP[split]
    cache_dir = get_cache_dir(split)
    val_cache_file = cache_dir / "validation_queries.jsonl"
    eval_cache_file = cache_dir / "evaluation_queries.jsonl"

    # Fast path: both cache files exist
    if val_cache_file.exists() and eval_cache_file.exists():
        def _read(f: Path) -> List[Dict]:
            queries = [json.loads(line) for line in f.read_text().splitlines() if line]
            print(f"✓ Loading cached {f.name} from {f}  ({len(queries)} queries)")
            return queries
        return _read(val_cache_file), _read(eval_cache_file)

    # Download path: fetch both CRUMB splits, pool, then apply our split JSON
    def _download_crumb_split(crumb_split_name: str) -> List[Dict]:
        print(f"Downloading {crumb_split_name} for {split}...")
        dataset = load_dataset("jfkback/crumb", crumb_split_name, split=crumb_name)
        queries = []
        for item in dataset:
            qrels = item.get("full_document_qrels") or item.get("passage_qrels") or []
            relevant_ids = [q["id"] for q in qrels if q.get("label", 0) > 0]
            if relevant_ids:
                queries.append({
                    "query_id": item["query_id"],
                    "query_content": item["query_content"],
                    "relevant_doc_ids": relevant_ids,
                })
        return queries

    all_queries: List[Dict] = []
    for crumb_split_name in ("validation_queries", "evaluation_queries"):
        all_queries += _download_crumb_split(crumb_split_name)

    # Dedup, then apply the hardcoded split
    seen: dict = {}
    for q in all_queries:
        seen[q["query_id"]] = q
    all_queries = list(seen.values())

    if n_queries and n_queries < len(all_queries):
        all_queries = all_queries[:n_queries]

    val_queries, eval_queries = _apply_split(all_queries, split)

    print(f"Caching {len(val_queries)} validation_queries to {val_cache_file}")
    with val_cache_file.open("w") as f:
        for q in val_queries:
            f.write(json.dumps(q) + "\n")

    print(f"Caching {len(eval_queries)} evaluation_queries to {eval_cache_file}")
    with eval_cache_file.open("w") as f:
        for q in eval_queries:
            f.write(json.dumps(q) + "\n")

    return val_queries, eval_queries


def load_full_corpus_streaming(split: str, max_docs: Optional[int] = None) -> List[Dict]:
    """
    Download corpus using STREAMING to avoid downloading all splits.

    Args:
        split: CRUMB split name
        max_docs: Maximum number of documents to download (None = all)
    """
    crumb_name = SPLIT_MAP[split]
    cache_dir = get_cache_dir(split)
    cache_file = cache_dir / "documents.jsonl"

    if cache_file.exists():
        print(f"✓ Loading cached corpus from {cache_file}")
        with cache_file.open("r") as f:
            docs = [json.loads(line) for line in f]
        print(f"  Loaded {len(docs)} documents")
        return docs

    if max_docs:
        print(f"Downloading corpus for {split} (streaming mode, limit={max_docs:,})...")
    else:
        print(f"Downloading corpus for {split} (streaming mode)...")

    # Use streaming=True to only fetch this split
    corpus_dataset = load_dataset(
        "jfkback/crumb",
        "full_document_corpus",
        split=crumb_name,
        streaming=True,
    )

    docs = []
    print("  Streaming documents...")

    # Use tqdm for progress if available
    try:
        pbar = tqdm(total=max_docs, unit=" docs")
        for item in corpus_dataset:
            docs.append({
                "doc_id": str(item["document_id"]),
                "text": item["document_content"],
                "metadata": {},
            })
            pbar.update(1)

            if max_docs and len(docs) >= max_docs:
                break
        pbar.close()
    except ImportError:
        for item in corpus_dataset:
            docs.append({
                "doc_id": str(item["document_id"]),
                "text": item["document_content"],
                "metadata": {},
            })
            if len(docs) % 10000 == 0:
                print(f"    Downloaded {len(docs):,} documents...")

            if max_docs and len(docs) >= max_docs:
                break

    print(f"\n  Total downloaded: {len(docs):,} documents")

    print(f"Caching to {cache_file}")
    with cache_file.open("w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    return docs


def main():
    parser = argparse.ArgumentParser(description="Download CRUMB corpus + queries (STREAMING)")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=list(SPLIT_MAP.keys()),
        help="CRUMB split name (omit if using --all)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all splits",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help="Limit number of queries (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to download (default: all)"
    )

    args = parser.parse_args()

    if not args.all and not args.split:
        parser.error("Either --split <name> or --all is required.")

    splits = list(SPLIT_MAP.keys()) if args.all else [args.split]

    for split in splits:
        cache_dir = get_cache_dir(split)
        print(f"\n{'='*70}")
        print(f"CRUMB Data Download (Streaming) — Split: {split}")
        if args.limit:
            print(f"Document limit: {args.limit:,}")
        print(f"Cache directory: {cache_dir}")
        print(f"{'='*70}\n")

        val_queries, eval_queries = load_queries(split, args.n_queries)
        docs = load_full_corpus_streaming(split, args.limit)

        val_relevant_ids = {str(rid) for q in val_queries for rid in q["relevant_doc_ids"]}
        eval_relevant_ids = {str(rid) for q in eval_queries for rid in q["relevant_doc_ids"]}
        print(f"\n{'='*70}")
        print(f"✓ Download complete")
        print(f"  Split       : {split}")
        print(f"  Val Queries : {len(val_queries)} ({len(val_relevant_ids)} relevant docs)")
        print(f"  Eval Queries: {len(eval_queries)} ({len(eval_relevant_ids)} relevant docs)")
        print(f"  Total docs  : {len(docs):,}")
        print(f"  Location    : {cache_dir}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
