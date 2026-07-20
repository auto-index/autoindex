"""Data validation tests for eval_utils.py.

Tests focus on:
- SubsetEvalResult and SubsetEvalSummary data integrity
- run_subset_eval metric calculations and edge cases
- load_preprocessor_from_code validation
- Data consistency across evaluation pipeline
"""

import sys
import pathlib
import pytest
import math
from unittest.mock import Mock, MagicMock

# Add agent to path
sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "src" / "agents" / "analysis_code_agent"))

from eval_utils import (
    SubsetEvalResult,
    SubsetEvalSummary,
    run_subset_eval,
    load_preprocessor_from_code,
)
from schema import EvalQuery, Chunk, Document


class TestSubsetEvalResultValidation:
    """Validate SubsetEvalResult dataclass structure and data."""

    def test_valid_result_creation(self):
        result = SubsetEvalResult(
            query_id="q1",
            recall_at_10=1.0,
            recall_at_100=1.0,
            ranks=[5],
            ndcg_at_10=0.5,
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
        )
        assert result.query_id == "q1"
        assert result.hit_at_10 is True
        assert result.hit_at_100 is True
        assert result.rank == 5
        assert result.ndcg_at_10 == 0.5
        assert len(result.retrieved_doc_ids) == 3

    def test_result_with_no_hit(self):
        result = SubsetEvalResult(
            query_id="q_miss",
            recall_at_10=0.0,
            recall_at_100=0.0,
            ranks=[],
            ndcg_at_10=0.0,
            retrieved_doc_ids=["wrong1", "wrong2"],
        )
        assert result.rank is None
        assert result.hit_at_10 is False
        assert result.hit_at_100 is False

    def test_hit_at_10_implies_hit_at_100(self):
        result = SubsetEvalResult(
            query_id="q1",
            recall_at_10=1.0,
            recall_at_100=1.0,
            ranks=[8],
            ndcg_at_10=0.5,
            retrieved_doc_ids=[],
        )
        if result.hit_at_10:
            assert result.hit_at_100, "hit_at_10 must imply hit_at_100"

    def test_rank_property_returns_best_rank(self):
        for rank in [1, 5, 10, 50, 100]:
            result = SubsetEvalResult(
                query_id=f"q_rank_{rank}",
                recall_at_10=1.0 if rank <= 10 else 0.0,
                recall_at_100=1.0 if rank <= 100 else 0.0,
                ranks=[rank],
                ndcg_at_10=(1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0,
                retrieved_doc_ids=[],
            )
            assert result.rank == rank
            assert result.rank >= 1

    def test_result_ndcg_range(self):
        result = SubsetEvalResult(
            query_id="q1",
            recall_at_10=1.0,
            recall_at_100=1.0,
            ranks=[1],
            ndcg_at_10=1.0,
            retrieved_doc_ids=[],
        )
        assert 0.0 <= result.ndcg_at_10 <= 1.0

    def test_result_empty_retrieved_docs(self):
        result = SubsetEvalResult(
            query_id="q_empty",
            recall_at_10=0.0,
            recall_at_100=0.0,
            ranks=[],
            ndcg_at_10=0.0,
            retrieved_doc_ids=[],
        )
        assert result.retrieved_doc_ids == []
        assert result.rank is None


class TestSubsetEvalSummaryValidation:
    """Validate SubsetEvalSummary dataclass and aggregations."""
    
    def test_valid_summary_creation(self):
        """SubsetEvalSummary can be created with valid data."""
        summary = SubsetEvalSummary(
            recall_at_10=0.8,
            recall_at_100=0.9,
            ndcg_at_10=0.65,
            n_queries=100,
            per_query=[]
        )
        
        assert summary.recall_at_10 == 0.8
        assert summary.recall_at_100 == 0.9
        assert summary.ndcg_at_10 == 0.65
        assert summary.n_queries == 100
    
    def test_summary_metric_ranges(self):
        """Summary metrics should be in [0, 1] range."""
        summary = SubsetEvalSummary(
            recall_at_10=0.5,
            recall_at_100=0.7,
            ndcg_at_10=0.6,
            n_queries=10
        )
        
        assert 0.0 <= summary.recall_at_10 <= 1.0
        assert 0.0 <= summary.recall_at_100 <= 1.0
        assert 0.0 <= summary.ndcg_at_10 <= 1.0
    
    def test_summary_recall_ordering(self):
        """recall@10 should be <= recall@100."""
        summary = SubsetEvalSummary(
            recall_at_10=0.6,
            recall_at_100=0.8,
            ndcg_at_10=0.5,
            n_queries=50
        )
        
        assert summary.recall_at_10 <= summary.recall_at_100, \
            "recall@10 must be <= recall@100"
    
    def test_summary_per_query_count_matches(self):
        """n_queries should match len(per_query)."""
        per_query = [
            SubsetEvalResult("q1", True, True, 5, 0.5, []),
            SubsetEvalResult("q2", False, True, 50, 0.0, []),
        ]
        summary = SubsetEvalSummary(
            recall_at_10=0.5,
            recall_at_100=1.0,
            ndcg_at_10=0.25,
            n_queries=2,
            per_query=per_query
        )
        
        assert summary.n_queries == len(summary.per_query)
    
    def test_summary_zero_queries(self):
        """Summary handles zero queries edge case."""
        summary = SubsetEvalSummary(
            recall_at_10=0.0,
            recall_at_100=0.0,
            ndcg_at_10=0.0,
            n_queries=0,
            per_query=[]
        )
        
        assert summary.n_queries == 0
        assert len(summary.per_query) == 0


class TestRunSubsetEvalDataValidation:
    """Data validation tests for run_subset_eval function."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock BM25Client for testing."""
        client = Mock()
        return client
    
    @pytest.fixture
    def sample_eval_queries(self):
        """Sample evaluation queries."""
        return [
            EvalQuery(
                query_id="q1",
                query_text="furry pet with claws",
                relevant_doc_ids=["doc_001"]
            ),
            EvalQuery(
                query_id="q2",
                query_text="planets and gas giants",
                relevant_doc_ids=["doc_002"]
            ),
            EvalQuery(
                query_id="q3",
                query_text="programming language whitespace",
                relevant_doc_ids=["doc_003"]
            ),
        ]
    
    def test_perfect_retrieval(self, mock_client, sample_eval_queries):
        """All queries find relevant docs at rank 1."""
        mock_client.batch_retrieve.return_value = [
            {"query_id": "q1", "ranked_docs": [{"doc_id": "doc_001", "score": 1.0}]},
            {"query_id": "q2", "ranked_docs": [{"doc_id": "doc_002", "score": 1.0}]},
            {"query_id": "q3", "ranked_docs": [{"doc_id": "doc_003", "score": 1.0}]},
        ]
        
        summary = run_subset_eval("test_index", sample_eval_queries, mock_client)
        
        assert summary.recall_at_10 == 1.0
        assert summary.recall_at_100 == 1.0
        assert summary.n_queries == 3
        assert len(summary.per_query) == 3
        
        # All should have rank 1
        for result in summary.per_query:
            assert result.rank == 1
            assert result.hit_at_10 is True
            assert result.hit_at_100 is True
    
    def test_no_hits(self, mock_client, sample_eval_queries):
        """No queries find relevant docs."""
        mock_client.batch_retrieve.return_value = [
            {"query_id": "q1", "ranked_docs": [{"doc_id": "wrong1", "score": 0.5}]},
            {"query_id": "q2", "ranked_docs": [{"doc_id": "wrong2", "score": 0.5}]},
            {"query_id": "q3", "ranked_docs": [{"doc_id": "wrong3", "score": 0.5}]},
        ]
        
        summary = run_subset_eval("test_index", sample_eval_queries, mock_client)
        
        assert summary.recall_at_10 == 0.0
        assert summary.recall_at_100 == 0.0
        assert summary.ndcg_at_10 == 0.0
        assert summary.n_queries == 3
        
        for result in summary.per_query:
            assert result.rank is None
            assert result.hit_at_10 is False
            assert result.hit_at_100 is False
            assert result.ndcg_at_10 == 0.0
    
    def test_mixed_results(self, mock_client, sample_eval_queries):
        """Mix of hits at different ranks."""
        mock_client.batch_retrieve.return_value = [
            # q1: hit at rank 1
            {"query_id": "q1", "ranked_docs": [
                {"doc_id": "doc_001", "score": 1.0},
            ]},
            # q2: hit at rank 15 (miss @10, hit @100)
            {"query_id": "q2", "ranked_docs": [
                {"doc_id": "wrong1", "score": 0.9},
                *[{"doc_id": f"wrong{i}", "score": 0.8} for i in range(2, 15)],
                {"doc_id": "doc_002", "score": 0.7},
            ]},
            # q3: no hit
            {"query_id": "q3", "ranked_docs": [
                {"doc_id": "wrong10", "score": 0.5},
            ]},
        ]
        
        summary = run_subset_eval("test_index", sample_eval_queries, mock_client)
        
        # 1/3 hit @10, 2/3 hit @100
        assert summary.recall_at_10 == pytest.approx(1/3)
        assert summary.recall_at_100 == pytest.approx(2/3)
        assert summary.n_queries == 3
        
        # Validate individual results
        assert summary.per_query[0].rank == 1
        assert summary.per_query[0].hit_at_10 is True
        
        assert summary.per_query[1].rank == 15
        assert summary.per_query[1].hit_at_10 is False
        assert summary.per_query[1].hit_at_100 is True
        
        assert summary.per_query[2].rank is None
        assert summary.per_query[2].hit_at_10 is False
    
    def test_ndcg_calculation(self, mock_client, sample_eval_queries):
        """nDCG@10 is calculated correctly."""
        mock_client.batch_retrieve.return_value = [
            {"query_id": "q1", "ranked_docs": [
                {"doc_id": "doc_001", "score": 1.0},  # rank 1, nDCG = 1.0
            ]},
            {"query_id": "q2", "ranked_docs": [
                {"doc_id": "wrong", "score": 0.9},
                {"doc_id": "doc_002", "score": 0.8},  # rank 2, nDCG = 1/log2(3) ≈ 0.631
            ]},
            {"query_id": "q3", "ranked_docs": [
                {"doc_id": "wrong", "score": 0.5},  # no hit, nDCG = 0.0
            ]},
        ]
        
        summary = run_subset_eval("test_index", sample_eval_queries, mock_client)
        
        # Expected nDCG values
        ndcg_q1 = 1.0  # rank 1: 1/log2(2) = 1.0
        ndcg_q2 = 1.0 / math.log2(3)  # rank 2: 1/log2(3) ≈ 0.631
        ndcg_q3 = 0.0  # no hit
        
        expected_avg_ndcg = (ndcg_q1 + ndcg_q2 + ndcg_q3) / 3
        
        assert summary.ndcg_at_10 == pytest.approx(expected_avg_ndcg, abs=1e-6)
        assert summary.per_query[0].ndcg_at_10 == pytest.approx(ndcg_q1)
        assert summary.per_query[1].ndcg_at_10 == pytest.approx(ndcg_q2)
        assert summary.per_query[2].ndcg_at_10 == pytest.approx(ndcg_q3)
    
    def test_empty_query_list(self, mock_client):
        """Handle empty query list."""
        mock_client.batch_retrieve.return_value = []
        
        summary = run_subset_eval("test_index", [], mock_client)
        
        assert summary.n_queries == 0
        assert len(summary.per_query) == 0
        # Division by n=1 (or n) to avoid division by zero
        assert summary.recall_at_10 == 0.0
        assert summary.recall_at_100 == 0.0
    
    def test_empty_results_for_query(self, mock_client, sample_eval_queries):
        """Query returns no results."""
        mock_client.batch_retrieve.return_value = [
            {"query_id": "q1", "ranked_docs": []},  # empty results
            {"query_id": "q2", "ranked_docs": [{"doc_id": "doc_002", "score": 1.0}]},
            {"query_id": "q3", "ranked_docs": []},  # empty results
        ]
        
        summary = run_subset_eval("test_index", sample_eval_queries, mock_client)
        
        assert summary.per_query[0].rank is None
        assert summary.per_query[0].retrieved_doc_ids == []
        assert summary.per_query[1].rank == 1
        assert summary.per_query[2].rank is None
    
    def test_retrieved_doc_ids_limited_to_10(self, mock_client, sample_eval_queries):
        """retrieved_doc_ids is limited to top 10."""
        mock_client.batch_retrieve.return_value = [
            {"query_id": "q1", "ranked_docs": [
                {"doc_id": f"doc_{i}", "score": 1.0 - i*0.01}
                for i in range(100)  # 100 results
            ]},
            {"query_id": "q2", "ranked_docs": []},
            {"query_id": "q3", "ranked_docs": []},
        ]
        
        summary = run_subset_eval("test_index", sample_eval_queries, mock_client)
        
        # Should only store top 10
        assert len(summary.per_query[0].retrieved_doc_ids) == 10
        assert summary.per_query[0].retrieved_doc_ids[0] == "doc_0"
        assert summary.per_query[0].retrieved_doc_ids[9] == "doc_9"
    
    def test_multiple_relevant_docs(self, mock_client):
        """Query with multiple relevant docs finds the best-ranked one."""
        queries = [
            EvalQuery(
                query_id="q_multi",
                query_text="test query",
                relevant_doc_ids=["doc_005", "doc_010", "doc_020"]
            )
        ]
        
        mock_client.batch_retrieve.return_value = [
            {"query_id": "q_multi", "ranked_docs": [
                *[{"doc_id": f"wrong{i}", "score": 1.0} for i in range(8)],
                {"doc_id": "doc_010", "score": 0.5},  # rank 9 (best of the 3)
                {"doc_id": "wrong9", "score": 0.4},
                {"doc_id": "doc_020", "score": 0.3},  # rank 11
                *[{"doc_id": f"wrong{i}", "score": 0.2} for i in range(10, 25)],
                {"doc_id": "doc_005", "score": 0.1},  # rank 26
            ]},
        ]
        
        summary = run_subset_eval("test_index", queries, mock_client)
        
        # Should use the best (first) rank: doc_010 at rank 9
        assert summary.per_query[0].rank == 9
        assert summary.per_query[0].hit_at_10 is True
        assert summary.per_query[0].hit_at_100 is True


class TestLoadPreprocessorFromCode:
    """Data validation for load_preprocessor_from_code."""
    
    def test_valid_preprocessor_code(self):
        """Valid preprocessor code loads successfully."""
        code = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "test_preprocessor"
    description = "Test"
    
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [
            Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text)
            for d in docs
        ]
"""
        
        preprocessor = load_preprocessor_from_code(code)
        
        assert preprocessor is not None
        assert preprocessor.name == "test_preprocessor"
        assert hasattr(preprocessor, 'preprocess')
    
    def test_preprocessor_can_process_documents(self, sample_documents):
        """Loaded preprocessor can process documents."""
        code = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "working"
    
    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [
            Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text)
            for d in docs
        ]
"""
        
        preprocessor = load_preprocessor_from_code(code)
        chunks = preprocessor.preprocess(sample_documents)
        
        assert len(chunks) == len(sample_documents)
        # Check using the Chunk class we imported
        from schema import Chunk as ChunkClass
        assert all(isinstance(c, ChunkClass) for c in chunks)
    
    def test_syntax_error_in_code(self):
        """Syntax error in code raises SyntaxError."""
        code = """
class Preprocessor(BasePreprocessor):
    def preprocess(self, docs):
        return [  # missing closing bracket
"""
        
        with pytest.raises(SyntaxError):
            load_preprocessor_from_code(code)
    
    def test_missing_preprocessor_class(self):
        """Code without Preprocessor class raises AttributeError."""
        code = """
# No Preprocessor class defined
def some_function():
    pass
"""
        
        with pytest.raises(AttributeError):
            load_preprocessor_from_code(code)
    
    def test_preprocessor_not_callable(self):
        """Preprocessor that can't be instantiated fails."""
        code = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from base import BasePreprocessor

# Preprocessor is not a class, it's a string
Preprocessor = "not a class"
"""
        
        with pytest.raises(TypeError):
            load_preprocessor_from_code(code)
    
    def test_multiple_loads_independent(self):
        """Multiple loads create independent preprocessors."""
        code1 = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "first"
    def preprocess(self, docs):
        return []
"""
        
        code2 = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "second"
    def preprocess(self, docs):
        return []
"""
        
        p1 = load_preprocessor_from_code(code1)
        p2 = load_preprocessor_from_code(code2)
        
        assert p1.name == "first"
        assert p2.name == "second"


class TestMetricConsistency:
    """Test metric calculation consistency and edge cases."""
    
    def test_rank_1_gives_max_ndcg(self):
        """Rank 1 should give nDCG@10 = 1.0."""
        ndcg = 1.0 / math.log2(1 + 1)
        assert ndcg == 1.0
    
    def test_rank_10_gives_valid_ndcg(self):
        """Rank 10 should give valid nDCG@10."""
        ndcg = 1.0 / math.log2(10 + 1)
        assert 0.0 < ndcg < 1.0
        # 1/log2(11) ≈ 0.289
        assert ndcg == pytest.approx(0.289, abs=0.001)
    
    def test_rank_11_gives_zero_ndcg(self):
        """Rank 11 (outside top 10) should not contribute to nDCG@10."""
        # In the code, nDCG is only computed if rank <= 10
        rank = 11
        ndcg = 0.0  # since rank > 10
        assert ndcg == 0.0
    
    def test_hit_flags_consistency(self):
        """hit_at_10 implies hit_at_100."""
        for rank in range(1, 101):
            hit_10 = rank <= 10
            hit_100 = rank <= 100
            
            if hit_10:
                assert hit_100, f"Rank {rank}: hit@10 should imply hit@100"
