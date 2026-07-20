"""
base.py – Abstract base class for all agent preprocessors.

Agents import this and subclass it, naming their class `Preprocessor`.
The eval harness discovers and validates agents via this interface.
"""

from __future__ import annotations

import sys
import pathlib

# Make schema importable when this file is imported from anywhere
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from abc import ABC, abstractmethod
from typing import List

from schema import Document, Chunk


class BasePreprocessor(ABC):
    """
    All agents must subclass this and name their concrete class `Preprocessor`.

    Class attributes (recommended, used in eval reports):
        name:        Short identifier, e.g. "sentence_chunker"
        description: One-line description of the strategy

    Required:
        preprocess(docs) -> List[Chunk]
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Transform raw documents into chunks ready for BM25 indexing.

        Contracts:
          - Return at least one Chunk per Document.
          - chunk.doc_id must match the source document's doc_id.
          - chunk_id must be globally unique across all returned chunks.
        """
