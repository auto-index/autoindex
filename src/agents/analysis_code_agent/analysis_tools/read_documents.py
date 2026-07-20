#!/usr/bin/env python3
"""Read document text snippets by doc_id from the tip_of_the_tongue corpus.

Usage examples:
  python src/agents/analysis_code_agent/analysis_tools/read_documents.py --doc-ids 12095072:0 12095072:1
  python src/agents/analysis_code_agent/analysis_tools/read_documents.py --doc-ids-file /tmp/doc_ids.txt --chars 1200
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable


def _project_root() -> pathlib.Path:
    # .../src/agents/analysis_code_agent/analysis_tools/read_documents.py -> project root
    return pathlib.Path(__file__).resolve().parents[4]


def _load_requested_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.doc_ids:
        ids.extend(args.doc_ids)
    if args.doc_ids_file:
        file_path = pathlib.Path(args.doc_ids_file)
        if not file_path.exists():
            raise FileNotFoundError(f"doc-ids file not found: {file_path}")
        for line in file_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                ids.append(value)
    # preserve order, drop duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def _iter_documents(documents_path: pathlib.Path) -> Iterable[dict]:
    with documents_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read doc text by doc_id from documents.jsonl")
    parser.add_argument("--doc-ids", nargs="*", default=[], help="One or more doc IDs to read")
    parser.add_argument("--doc-ids-file", default="", help="Optional newline-delimited file containing doc IDs")
    parser.add_argument("--split", default="tip_of_the_tongue", help="Data split under data/ (default: tip_of_the_tongue)")
    parser.add_argument("--chars", type=int, default=800, help="Max characters to print per document")
    parser.add_argument("--show-metadata", action="store_true", help="Print metadata JSON if present")
    args = parser.parse_args()

    requested_ids = _load_requested_ids(args)
    if not requested_ids:
        print("No doc IDs provided. Use --doc-ids and/or --doc-ids-file.", file=sys.stderr)
        return 2

    root = _project_root()
    docs_path = root / "data" / args.split / "documents.jsonl"
    if not docs_path.exists():
        # fall back to root data/ path used by some scripts
        docs_path = root / "data" / "documents.jsonl"
    if not docs_path.exists():
        print(f"Could not find documents.jsonl for split '{args.split}'.", file=sys.stderr)
        return 2

    requested_set = set(requested_ids)
    found: dict[str, dict] = {}

    for doc in _iter_documents(docs_path):
        doc_id = doc.get("doc_id")
        if doc_id in requested_set:
            found[doc_id] = doc
            if len(found) == len(requested_set):
                break

    for doc_id in requested_ids:
        print(f"DOC: {doc_id}")
        doc = found.get(doc_id)
        if doc is None:
            print("<MISSING>")
            print("----")
            continue

        text = (doc.get("text") or "").replace("\n", " ")
        print(text[: max(args.chars, 0)])
        if args.show_metadata:
            metadata = doc.get("metadata", {})
            print("METADATA:", json.dumps(metadata, ensure_ascii=False))
        print("----")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
