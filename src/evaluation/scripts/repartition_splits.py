#!/usr/bin/env python3
"""
Repartition validation/evaluation query JSONL files to match query_splits.json.
Run after editing query_splits.json to avoid a full re-download.

Usage:
    uv run python src/evaluation/scripts/repartition_splits.py
    uv run python src/evaluation/scripts/repartition_splits.py --split tip_of_the_tongue
    uv run python src/evaluation/scripts/repartition_splits.py --dry-run
"""
import argparse, json, pathlib, sys

PROJECT_ROOT = pathlib.Path(__file__).parents[3]
DATA_DIR     = PROJECT_ROOT / "data"
SPLITS_JSON  = PROJECT_ROOT / "query_splits.json"


def load_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def repartition(split_name, val_ids, eval_ids, dry_run=False):
    split_dir = DATA_DIR / split_name
    val_file  = split_dir / "validation_queries.jsonl"
    eval_file = split_dir / "evaluation_queries.jsonl"

    if not split_dir.exists():
        print(f"  SKIP: data/{split_name}/ not found", file=sys.stderr)
        return

    all_queries = {}
    for f in [val_file, eval_file]:
        if f.exists():
            for q in load_jsonl(f):
                all_queries[q["query_id"]] = q

    for qid in val_ids + eval_ids:
        if qid not in all_queries:
            print(f"  WARNING: query_id {qid!r} not found in data files — skipping", file=sys.stderr)

    new_val  = [all_queries[qid] for qid in val_ids  if qid in all_queries]
    new_eval = [all_queries[qid] for qid in eval_ids if qid in all_queries]

    old_val_count  = len(load_jsonl(val_file))  if val_file.exists()  else 0
    old_eval_count = len(load_jsonl(eval_file)) if eval_file.exists() else 0
    print(f"  validation:  {old_val_count} -> {len(new_val)}")
    print(f"  evaluation:  {old_eval_count} -> {len(new_eval)}")

    if not dry_run:
        for path, rows in [(val_file, new_val), (eval_file, new_eval)]:
            with path.open("w") as f:
                for q in rows:
                    f.write(json.dumps(q) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", help="Process only this split (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    config = json.loads(SPLITS_JSON.read_text())
    targets = {args.split: config[args.split]} if args.split else config

    for split_name, ids in targets.items():
        print(f"\n[{split_name}]")
        repartition(split_name, ids["validation"], ids["evaluation"], dry_run=args.dry_run)

    if args.dry_run:
        print("\n(dry run — no files written)")


if __name__ == "__main__":
    main()
