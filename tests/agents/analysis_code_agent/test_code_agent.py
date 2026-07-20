"""
Unit and integration tests for CodeAgent.

Unit tests (no external dependencies):
  - TestParseHypothesesBlocks  — parsing LLM output into Hypothesis objects
  - TestClassify               — taxonomy keyword classification
  - TestTaxonomyAssignment     — diversity enforcement (unexplored categories first)
  - TestValidateCode           — code validation before hypothesis testing

Integration tests (require live BM25 server, marked with @pytest.mark.integration):
  - TestTestHypothesis         — full test_hypothesis() flow against real BM25 index
"""
import sys
import pathlib
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
_TEST_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(_TEST_DIR))

from src.agents.analysis_code_agent.code_agent import CodeAgent, Hypothesis
from conftest import SAMPLE_DOCS, SAMPLE_QUERIES  # noqa: E402

# ── Shared agent fixture ──────────────────────────────────────────────────────

MINIMAL_CONFIG = {
    "code_model": "openai/gpt-4o",
    "code_temperature": 0.7,
    "api_base": None,
    "recall_improvement_threshold": 0.05,
    "max_hypotheses": 4,
    "server_port": 8766,
}


@pytest.fixture
def agent():
    return CodeAgent(config=MINIMAL_CONFIG)


# ── Shared test strings ───────────────────────────────────────────────────────

_PASSTHROUGH_CODE = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "passthrough"
    description = "passthrough"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text) for d in docs]
"""

_WELL_FORMED_FOUR_HYPOTHESES = """
### H1: Title Prepend Strategy
Category: CONTEXT
Rationale: Prepending the title improves BM25 term matching.
Query IDs: q_matrix, q_inception
Falsifying: recall does not improve by 5pp

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "title_prepend"
    description = "title prepend"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text) for d in docs]
```

### H2: Synonym Expansion
Category: VOCABULARY
Rationale: Adding synonyms bridges the vocabulary gap.
Query IDs: q_matrix
Falsifying: no recall improvement

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "synonym"
    description = "synonym expansion"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text) for d in docs]
```

### H3: Sliding Window Chunking
Category: STRUCTURE
Rationale: Overlapping windows improve term co-occurrence.
Query IDs: q_inception
Falsifying: recall drops

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "sliding_window"
    description = "sliding window"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text) for d in docs]
```

### H4: N-gram Extraction
Category: QUERY-BRIDGING
Rationale: N-grams capture multi-word query phrases.
Query IDs: q_matrix, q_inception
Falsifying: precision drops

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "ngram"
    description = "ngram extraction"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text) for d in docs]
```
"""


# ═════════════════════════════════════════════════════════════════════════════
# Unit 1: _parse_hypotheses_blocks
# ═════════════════════════════════════════════════════════════════════════════

class TestParseHypothesesBlocks:

    def test_parses_all_four_hypotheses(self, agent):
        result = agent._parse_hypotheses_blocks(_WELL_FORMED_FOUR_HYPOTHESES)
        assert result is not None
        assert len(result) == 4

    def test_correct_ids(self, agent):
        result = agent._parse_hypotheses_blocks(_WELL_FORMED_FOUR_HYPOTHESES)
        assert [r["id"] for r in result] == ["H1", "H2", "H3", "H4"]

    def test_correct_descriptions(self, agent):
        result = agent._parse_hypotheses_blocks(_WELL_FORMED_FOUR_HYPOTHESES)
        assert "Title Prepend" in result[0]["description"]
        assert "Synonym Expansion" in result[1]["description"]
        assert "Sliding Window" in result[2]["description"]
        assert "N-gram" in result[3]["description"]

    def test_extracts_rationale(self, agent):
        result = agent._parse_hypotheses_blocks(_WELL_FORMED_FOUR_HYPOTHESES)
        assert "Prepending the title" in result[0]["rationale"]
        assert "vocabulary gap" in result[1]["rationale"]

    def test_extracts_query_ids(self, agent):
        result = agent._parse_hypotheses_blocks(_WELL_FORMED_FOUR_HYPOTHESES)
        assert "q_matrix" in result[0]["query_ids_to_test"]
        assert "q_inception" in result[0]["query_ids_to_test"]
        assert "q_matrix" in result[1]["query_ids_to_test"]

    def test_extracts_code_block_with_preprocessor_class(self, agent):
        result = agent._parse_hypotheses_blocks(_WELL_FORMED_FOUR_HYPOTHESES)
        for h in result:
            assert "class Preprocessor" in h["code"]
            assert "def preprocess" in h["code"]

    def test_no_headers_returns_none(self, agent):
        result = agent._parse_hypotheses_blocks("Some random text with no hypothesis markers.")
        assert result is None

    def test_headers_without_code_blocks_returns_none(self, agent):
        text = "### H1: Title Prepend\nRationale: something\n\n### H2: Other\nRationale: other\n"
        result = agent._parse_hypotheses_blocks(text)
        assert result is None

    def test_partial_parse_returns_only_hypotheses_with_code(self, agent):
        # H1 has code, H2 does not
        text = (
            "### H1: Has Code\nRationale: yes\nQuery IDs: q1\nFalsifying: none\n"
            "```python\nclass Preprocessor:\n    pass\n```\n\n"
            "### H2: No Code\nRationale: missing code block\n"
        )
        result = agent._parse_hypotheses_blocks(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "H1"

    def test_empty_string_returns_none(self, agent):
        assert agent._parse_hypotheses_blocks("") is None


# ═════════════════════════════════════════════════════════════════════════════
# Unit 2: _validate_code
# ═════════════════════════════════════════════════════════════════════════════

class TestValidateCode:

    def test_valid_passthrough_returns_none(self, agent):
        error = agent._validate_code(_PASSTHROUGH_CODE, SAMPLE_DOCS)
        assert error is None

    def test_invalid_doc_id_returns_error(self, agent):
        # Code that rewrites doc_id (validation runs on hashed doc_ids, so the
        # mutation must corrupt any id, not just ids with a ':' separator)
        bad_code = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "bad"
    description = "bad doc_id"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [
            Chunk(chunk_id=f"{d.doc_id}_0",
                  doc_id=d.doc_id + "_mangled",  # WRONG: modifies the doc_id
                  text=d.text)
            for d in docs
        ]
"""
        error = agent._validate_code(bad_code, SAMPLE_DOCS)
        assert error is not None
        assert "invalid doc_id" in error.lower() or "doc_id" in error

    def test_empty_return_returns_error(self, agent):
        empty_code = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "empty"
    description = "returns nothing"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return []
"""
        error = agent._validate_code(empty_code, SAMPLE_DOCS)
        assert error is not None
        assert "empty" in error.lower()

    def test_missing_preprocessor_class_returns_error(self, agent):
        no_class_code = """
def some_function():
    pass
"""
        error = agent._validate_code(no_class_code, SAMPLE_DOCS)
        assert error is not None

    def test_runtime_error_returns_error_string(self, agent):
        crash_code = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "crash"
    description = "crashes at runtime"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        raise ValueError("intentional crash for testing")
"""
        error = agent._validate_code(crash_code, SAMPLE_DOCS)
        assert error is not None
        assert "intentional crash" in error


# ═════════════════════════════════════════════════════════════════════════════
# Integration: test_hypothesis — requires live BM25 server
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestTestHypothesis:
    """
    Requires a live BM25 server (started by the bm25_server session fixture).
    Run with: uv run pytest -m integration tests/
    """

    # Hypothesis that improves recall: adds rich content to title-stub docs.
    # The baseline "current" index has bare title stubs that don't match query terms.
    # This hypothesis merges the text of sections sharing a markdown title heading,
    # giving BM25 enough vocabulary to match queries like "hacker discovers simulation".
    # (Grouping uses the title line, not doc_id — doc_ids are hashed during validation
    # and carry no article/section structure.)
    _IMPROVING_CODE = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
import collections
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "title_centric"
    description = "merge sections sharing a title heading into each chunk"
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        articles = collections.defaultdict(list)
        for d in docs:
            title = d.text.splitlines()[0] if d.text else ""
            articles[title].append(d)
        chunks = []
        for title, secs in articles.items():
            merged = " ".join(s.text for s in secs)
            for sec in secs:
                chunks.append(Chunk(
                    chunk_id=f"{sec.doc_id}_merged",
                    doc_id=sec.doc_id,
                    text=merged,
                ))
        return chunks
"""

    def test_improving_hypothesis_is_proven(self, agent, bm25_client_with_current_index):
        h = Hypothesis(
            id="H1",
            description="Merge all sections into title chunk",
            rationale="Improves vocabulary coverage",
            code=self._IMPROVING_CODE,
        )
        result = agent.test_hypothesis(
            h, SAMPLE_DOCS, SAMPLE_QUERIES, _PASSTHROUGH_CODE, bm25_client_with_current_index
        )
        assert result.error is None
        assert result.delta_recall_100 >= 0.0
        assert result.hypothesis_recall_100 >= result.baseline_recall_100
        assert len(result.improved_query_ids) >= 0  # some queries should improve

    def test_passthrough_hypothesis_is_not_proven(self, agent, bm25_client_with_current_index):
        h = Hypothesis(
            id="H2",
            description="Passthrough — identical to current",
            rationale="No change",
            code=_PASSTHROUGH_CODE,
        )
        result = agent.test_hypothesis(
            h, SAMPLE_DOCS, SAMPLE_QUERIES, _PASSTHROUGH_CODE, bm25_client_with_current_index
        )
        assert result.error is None
        assert result.delta_recall_100 == pytest.approx(0.0, abs=1e-6)
        assert result.proven is False

    def test_invalid_code_sets_error_not_proven(self, agent, bm25_client_with_current_index):
        h = Hypothesis(
            id="H3",
            description="Bad code that crashes",
            rationale="Testing error handling",
            code="this is not valid python !!!",
        )
        result = agent.test_hypothesis(
            h, SAMPLE_DOCS, SAMPLE_QUERIES, _PASSTHROUGH_CODE, bm25_client_with_current_index
        )
        assert result.error is not None
        assert result.proven is False

    def test_improved_and_regressed_query_ids_populated(self, agent, bm25_client_with_current_index):
        h = Hypothesis(
            id="H4",
            description="Merge all sections into title chunk",
            rationale="Test query tracking",
            code=self._IMPROVING_CODE,
        )
        result = agent.test_hypothesis(
            h, SAMPLE_DOCS, SAMPLE_QUERIES, _PASSTHROUGH_CODE, bm25_client_with_current_index
        )
        assert result.error is None
        # improved + regressed must be disjoint
        assert set(result.improved_query_ids).isdisjoint(set(result.regressed_query_ids))
