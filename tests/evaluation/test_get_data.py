"""Data validation tests for get_data.py script.

Tests focus on:
- JSONL file format validation
- Document structure validation (doc_id, text, metadata)
- Query structure validation (query_id, query_content, relevant_doc_ids)
- Cache directory management
- Data integrity and consistency
- Edge cases (empty files, missing fields, corrupted data)
"""

import sys
import pathlib
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts to path
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "src" / "evaluation" / "scripts"))

from get_data import (
    get_cache_dir,
    load_queries,
    load_full_corpus_streaming,
    SPLIT_MAP,
)


class TestCacheDirectoryManagement:
    """Test cache directory creation and management."""
    
    def test_cache_dir_no_limit(self):
        """Cache directory created for a split."""
        cache_dir = get_cache_dir("tip_of_the_tongue")

        # Should create data/<split>/ directory
        assert "tip_of_the_tongue" in str(cache_dir)
        assert "docs" not in str(cache_dir)
    
    def test_cache_dir_creates_directory(self):
        """Cache directory is created if it doesn't exist."""
        cache_dir = get_cache_dir("tip_of_the_tongue")
        assert cache_dir.exists()
        assert cache_dir.is_dir()


class TestDocumentStructureValidation:
    """Validate document JSONL structure and fields."""
    
    def test_document_required_fields(self, tmp_path):
        """Documents have required fields: doc_id, text, metadata."""
        # Use a real split name
        split = "tip_of_the_tongue"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        docs = [
            {"doc_id": "d1", "text": "Document 1 text", "metadata": {}},
            {"doc_id": "d2", "text": "Document 2 text", "metadata": {"key": "value"}},
        ]
        
        with docs_file.open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split, max_docs=None)
        
        assert len(loaded_docs) == 2
        for doc in loaded_docs:
            assert "doc_id" in doc
            assert "text" in doc
            assert "metadata" in doc
            assert isinstance(doc["doc_id"], str)
            assert isinstance(doc["text"], str)
            assert isinstance(doc["metadata"], dict)
    
    def test_document_doc_id_not_empty(self, tmp_path):
        """Document doc_id should not be empty."""
        split = "stack_exchange"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        docs = [
            {"doc_id": "valid_id", "text": "text", "metadata": {}},
        ]
        
        with docs_file.open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
        
        for doc in loaded_docs:
            assert len(doc["doc_id"]) > 0
    
    def test_document_text_can_be_empty(self, tmp_path):
        """Document text can be empty (edge case)."""
        split = "paper_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        docs = [{"doc_id": "empty_text", "text": "", "metadata": {}}]
        
        with docs_file.open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
        
        assert loaded_docs[0]["text"] == ""
    
    def test_document_unicode_text(self, tmp_path):
        """Documents preserve unicode text."""
        split = "code_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        docs = [
            {"doc_id": "unicode", "text": "Café 日本語 🎉", "metadata": {}}
        ]
        
        with docs_file.open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
        
        assert "Café" in loaded_docs[0]["text"]
        assert "日本語" in loaded_docs[0]["text"]
        assert "🎉" in loaded_docs[0]["text"]


class TestQueryStructureValidation:
    """Validate query JSONL structure and fields."""
    
    def test_query_required_fields(self, tmp_path):
        """Queries have required fields: query_id, query_content, relevant_doc_ids."""
        split = "tip_of_the_tongue"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        queries_file = cache_dir / "validation_queries.jsonl"
        (cache_dir / "evaluation_queries.jsonl").touch()
        
        queries = [
            {
                "query_id": "q1",
                "query_content": "query text 1",
                "relevant_doc_ids": ["d1", "d2"]
            },
            {
                "query_id": "q2",
                "query_content": "query text 2",
                "relevant_doc_ids": ["d3"]
            },
        ]
        
        with queries_file.open("w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            val_queries, eval_queries = load_queries(split)
        loaded_queries = val_queries + eval_queries
        
        assert len(loaded_queries) == 2
        for query in loaded_queries:
            assert "query_id" in query
            assert "query_content" in query
            assert "relevant_doc_ids" in query
    
    def test_query_relevant_doc_ids_is_list(self, tmp_path):
        """relevant_doc_ids is always a list."""
        split = "legal_qa"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        queries_file = cache_dir / "validation_queries.jsonl"
        (cache_dir / "evaluation_queries.jsonl").touch()
        
        queries = [
            {"query_id": "q1", "query_content": "test", "relevant_doc_ids": ["d1"]},
            {"query_id": "q2", "query_content": "test", "relevant_doc_ids": []},
        ]
        
        with queries_file.open("w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            val_queries, eval_queries = load_queries(split)
        loaded_queries = val_queries + eval_queries
        
        for query in loaded_queries:
            assert isinstance(query["relevant_doc_ids"], list)
    
    def test_query_unicode_content(self, tmp_path):
        """Queries preserve unicode in query_content."""
        split = "clinical_trial"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        queries_file = cache_dir / "validation_queries.jsonl"
        (cache_dir / "evaluation_queries.jsonl").touch()
        
        queries = [
            {
                "query_id": "q_unicode",
                "query_content": "Café 日本語 🎉",
                "relevant_doc_ids": ["d1"]
            }
        ]
        
        with queries_file.open("w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            val_queries, eval_queries = load_queries(split)
        loaded_queries = val_queries + eval_queries
        
        assert "Café" in loaded_queries[0]["query_content"]
        assert "日本語" in loaded_queries[0]["query_content"]


class TestJSONLFormatValidation:
    """Test JSONL file format handling."""
    
    def test_empty_documents_file(self, tmp_path):
        """Empty documents.jsonl returns empty list."""
        split = "theorem_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        docs_file.touch()  # Create empty file
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
        
        assert loaded_docs == []
    
    def test_empty_queries_file(self, tmp_path):
        """Empty queries.jsonl returns empty list."""
        split = "set_operation_entity_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        queries_file = cache_dir / "validation_queries.jsonl"
        (cache_dir / "evaluation_queries.jsonl").touch()
        queries_file.touch()
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            val_queries, eval_queries = load_queries(split)
        loaded_queries = val_queries + eval_queries
        
        assert loaded_queries == []
    
    def test_malformed_json_line_handling(self, tmp_path):
        """Malformed JSON lines raise error."""
        split = "paper_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        # Write malformed JSON
        with docs_file.open("w") as f:
            f.write('{"doc_id": "d1", "text": "ok", "metadata": {}}\n')
            f.write('THIS IS NOT JSON\n')  # Malformed line
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            with pytest.raises(json.JSONDecodeError):
                load_full_corpus_streaming(split)


class TestDataIntegrityAndConsistency:
    """Test data consistency between queries and documents."""
    
    def test_query_references_existing_documents(self, tmp_path):
        """All query relevant_doc_ids reference actual documents."""
        split = "tip_of_the_tongue"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        
        # Create documents
        docs_file = cache_dir / "documents.jsonl"
        docs = [
            {"doc_id": "d1", "text": "Doc 1", "metadata": {}},
            {"doc_id": "d2", "text": "Doc 2", "metadata": {}},
            {"doc_id": "d3", "text": "Doc 3", "metadata": {}},
        ]
        with docs_file.open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        
        # Create queries referencing these docs
        queries_file = cache_dir / "validation_queries.jsonl"
        (cache_dir / "evaluation_queries.jsonl").touch()
        queries = [
            {"query_id": "q1", "query_content": "query 1", "relevant_doc_ids": ["d1"]},
            {"query_id": "q2", "query_content": "query 2", "relevant_doc_ids": ["d2", "d3"]},
        ]
        with queries_file.open("w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
            val_queries, eval_queries = load_queries(split)
        loaded_queries = val_queries + eval_queries
        
        # Validate consistency
        doc_ids = {doc["doc_id"] for doc in loaded_docs}
        for query in loaded_queries:
            for rel_id in query["relevant_doc_ids"]:
                assert rel_id in doc_ids, \
                    f"Query {query['query_id']} references missing doc {rel_id}"
    
    def test_document_ids_unique(self, tmp_path):
        """All document IDs are unique."""
        split = "stack_exchange"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        docs = [
            {"doc_id": "d1", "text": "Doc 1", "metadata": {}},
            {"doc_id": "d2", "text": "Doc 2", "metadata": {}},
            {"doc_id": "d3", "text": "Doc 3", "metadata": {}},
        ]
        
        with docs_file.open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
        
        doc_ids = [doc["doc_id"] for doc in loaded_docs]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_ids found"
    
    def test_query_ids_unique(self, tmp_path):
        """All query IDs are unique."""
        split = "code_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        queries_file = cache_dir / "validation_queries.jsonl"
        (cache_dir / "evaluation_queries.jsonl").touch()
        
        queries = [
            {"query_id": "q1", "query_content": "Query 1", "relevant_doc_ids": ["d1"]},
            {"query_id": "q2", "query_content": "Query 2", "relevant_doc_ids": ["d2"]},
            {"query_id": "q3", "query_content": "Query 3", "relevant_doc_ids": ["d3"]},
        ]
        
        with queries_file.open("w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            val_queries, eval_queries = load_queries(split)
        loaded_queries = val_queries + eval_queries
        
        query_ids = [q["query_id"] for q in loaded_queries]
        assert len(query_ids) == len(set(query_ids)), "Duplicate query_ids found"


class TestCacheBehavior:
    """Test caching behavior and file reuse."""
    
    def test_loads_from_cache_if_exists(self, tmp_path):
        """If cache file exists, load from cache instead of downloading."""
        split = "legal_qa"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        # Pre-populate cache
        cached_docs = [{"doc_id": "cached", "text": "From cache", "metadata": {}}]
        with docs_file.open("w") as f:
            for doc in cached_docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
        
        assert len(loaded_docs) == 1
        assert loaded_docs[0]["doc_id"] == "cached"
        assert loaded_docs[0]["text"] == "From cache"
    
    def test_queries_load_from_cache(self, tmp_path):
        """Queries load from cache if file exists."""
        split = "theorem_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        queries_file = cache_dir / "validation_queries.jsonl"
        (cache_dir / "evaluation_queries.jsonl").touch()
        
        cached_queries = [
            {"query_id": "cached_q", "query_content": "From cache", "relevant_doc_ids": ["d1"]}
        ]
        with queries_file.open("w") as f:
            for q in cached_queries:
                f.write(json.dumps(q) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            val_queries, eval_queries = load_queries(split)
        loaded_queries = val_queries + eval_queries
        
        assert len(loaded_queries) == 1
        assert loaded_queries[0]["query_id"] == "cached_q"


class TestSplitConfiguration:
    """Test split configuration and expected sizes."""
    
    def test_all_splits_have_mapping(self):
        """All splits in SPLIT_MAP are valid."""
        for split_name in SPLIT_MAP:
            assert isinstance(split_name, str)
            assert len(split_name) > 0
    


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_text_document(self, tmp_path):
        """Documents with very long text are handled."""
        split = "clinical_trial"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        # Create very long text
        long_text = "word " * 100000  # ~500KB
        docs = [{"doc_id": "long", "text": long_text, "metadata": {}}]
        
        with docs_file.open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
        
        assert len(loaded_docs[0]["text"]) > 400000
    
    def test_document_with_newlines_in_text(self, tmp_path):
        """Documents with newlines in text are preserved."""
        split = "paper_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        docs_file = cache_dir / "documents.jsonl"
        
        docs = [
            {"doc_id": "newlines", "text": "Line 1\nLine 2\n\nLine 3", "metadata": {}}
        ]
        
        with docs_file.open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            loaded_docs = load_full_corpus_streaming(split)
        
        assert "\n" in loaded_docs[0]["text"]
        assert loaded_docs[0]["text"].count("\n") == 3
    
    def test_query_with_special_characters(self, tmp_path):
        """Queries with special characters are preserved."""
        split = "set_operation_entity_retrieval"
        cache_dir = tmp_path / split
        cache_dir.mkdir()
        queries_file = cache_dir / "validation_queries.jsonl"
        (cache_dir / "evaluation_queries.jsonl").touch()
        
        queries = [
            {
                "query_id": "special",
                "query_content": "Query with @#$% & special <chars>",
                "relevant_doc_ids": ["d1"]
            }
        ]
        
        with queries_file.open("w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        
        with patch('get_data.get_cache_dir', return_value=cache_dir):
            val_queries, eval_queries = load_queries(split)
        loaded_queries = val_queries + eval_queries
        
        assert "@#$%" in loaded_queries[0]["query_content"]
        assert "<chars>" in loaded_queries[0]["query_content"]
