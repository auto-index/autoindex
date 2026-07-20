"""
schema.py – Shared data classes for the index-generation harness.

Agents import Document and Chunk from here to type their preprocess() function.
Test scripts import all three classes to load data and run evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    """A raw document from the corpus."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A preprocessed unit ready for indexing. chunk.doc_id links back to a Document."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalQuery:
    """An evaluation query with ground-truth relevant document IDs."""
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]
