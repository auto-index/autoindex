"""Tests for evaluation schema data classes.

Validates Document, Chunk, and EvalQuery dataclasses to ensure:
- Required fields are present
- Types are correct
- Edge cases are handled
- Relationships between objects are valid (e.g., Chunk.doc_id → Document.doc_id)
"""

import pytest
from schema import Document, Chunk, EvalQuery


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_document_valid_creation(self):
        """Document can be created with required fields."""
        doc = Document(
            doc_id="test_doc_123",
            text="This is a test document about machine learning."
        )
        
        assert doc.doc_id == "test_doc_123"
        assert doc.text == "This is a test document about machine learning."
        assert doc.metadata == {}
    
    def test_document_with_metadata(self):
        """Document can include custom metadata."""
        doc = Document(
            doc_id="wiki_paris",
            text="Paris is the capital of France.",
            metadata={"title": "Paris", "source": "wikipedia", "year": 2024}
        )
        
        assert doc.metadata["title"] == "Paris"
        assert doc.metadata["source"] == "wikipedia"
        assert doc.metadata["year"] == 2024
        assert len(doc.metadata) == 3
    
    def test_document_empty_text_allowed(self):
        """Document can have empty text (edge case, might be invalid data)."""
        doc = Document(doc_id="empty_doc", text="")
        assert doc.text == ""
        assert doc.doc_id == "empty_doc"
    
    def test_document_whitespace_text(self):
        """Document with only whitespace is valid (but might be bad data)."""
        doc = Document(doc_id="whitespace_doc", text="   \n\t  ")
        assert len(doc.text) > 0
        assert doc.text.strip() == ""
    
    def test_document_unicode_text(self):
        """Document text can contain unicode characters."""
        doc = Document(
            doc_id="unicode_doc",
            text="Café, naïve, 日本語, emoji: 🎉"
        )
        assert "Café" in doc.text
        assert "日本語" in doc.text
        assert "🎉" in doc.text
    
    def test_document_long_text(self):
        """Document can handle long text (typical Wikipedia article)."""
        long_text = "word " * 10000  # ~50KB
        doc = Document(doc_id="long_doc", text=long_text)
        assert len(doc.text) >= 50000


class TestChunk:
    """Tests for Chunk dataclass."""
    
    def test_chunk_valid_creation(self):
        """Chunk can be created with required fields."""
        chunk = Chunk(
            chunk_id="doc123_chunk_0",
            doc_id="doc123",
            text="First paragraph of the document."
        )
        
        assert chunk.chunk_id == "doc123_chunk_0"
        assert chunk.doc_id == "doc123"
        assert chunk.text == "First paragraph of the document."
        assert chunk.metadata == {}
    
    def test_chunk_with_metadata(self):
        """Chunk can include custom metadata."""
        chunk = Chunk(
            chunk_id="doc123_chunk_1",
            doc_id="doc123",
            text="Second paragraph.",
            metadata={
                "section": "Introduction",
                "position": 1,
                "tokens": 42
            }
        )
        
        assert chunk.metadata["section"] == "Introduction"
        assert chunk.metadata["position"] == 1
        assert chunk.metadata["tokens"] == 42
    
    def test_chunk_id_doc_id_relationship(self):
        """Chunk ID typically contains the doc_id (convention, not enforced)."""
        chunk = Chunk(
            chunk_id="wiki_paris_c0",
            doc_id="wiki_paris",
            text="Paris is the capital."
        )
        
        # Convention: chunk_id starts with doc_id
        assert chunk.chunk_id.startswith(chunk.doc_id)


class TestEvalQuery:
    """Tests for EvalQuery dataclass."""
    
    def test_eval_query_valid_creation(self):
        """EvalQuery can be created with required fields."""
        query = EvalQuery(
            query_id="q1",
            query_text="What is the capital of France?",
            relevant_doc_ids=["wiki_paris", "wiki_france"]
        )
        
        assert query.query_id == "q1"
        assert query.query_text == "What is the capital of France?"
        assert query.relevant_doc_ids == ["wiki_paris", "wiki_france"]
        assert len(query.relevant_doc_ids) == 2
    
    def test_eval_query_single_relevant_doc(self):
        """EvalQuery can have a single relevant document."""
        query = EvalQuery(
            query_id="q2",
            query_text="Eiffel Tower location",
            relevant_doc_ids=["wiki_eiffel_tower"]
        )
        
        assert len(query.relevant_doc_ids) == 1
        assert query.relevant_doc_ids[0] == "wiki_eiffel_tower"
    
    def test_eval_query_empty_relevant_docs(self):
        """EvalQuery can have empty relevant_doc_ids list (edge case)."""
        query = EvalQuery(
            query_id="q_no_results",
            query_text="Nonexistent topic",
            relevant_doc_ids=[]
        )
        
        assert query.relevant_doc_ids == []


class TestSchemaWithFixtures:
    """Tests using conftest.py fixtures."""
    
    def test_sample_documents_load(self, sample_documents):
        """Fixture documents load correctly."""
        assert len(sample_documents) == 8
        assert all(isinstance(doc, Document) for doc in sample_documents)
        assert sample_documents[0].doc_id == "doc_001"
        assert "cat" in sample_documents[0].text.lower()
    
    def test_sample_queries_load(self, sample_queries):
        """Fixture queries load correctly."""
        assert len(sample_queries) == 5
        assert all(isinstance(q, EvalQuery) for q in sample_queries)
        assert sample_queries[0].query_id == "q1"
        assert len(sample_queries[0].relevant_doc_ids) > 0
    
    def test_sample_chunks_created(self, sample_chunks):
        """Baseline chunks are created correctly."""
        assert len(sample_chunks) == 8
        assert all(isinstance(c, Chunk) for c in sample_chunks)
        assert all(c.chunk_id.endswith("_0") for c in sample_chunks)
    
    def test_query_doc_references_valid(self, sample_documents, sample_queries):
        """All query relevant_doc_ids reference actual documents."""
        doc_ids = {doc.doc_id for doc in sample_documents}
        
        for query in sample_queries:
            for rel_id in query.relevant_doc_ids:
                assert rel_id in doc_ids, \
                    f"Query {query.query_id} references missing doc {rel_id}"
    
    def test_chunks_reference_documents(self, sample_documents, sample_chunks):
        """All chunks reference valid documents."""
        doc_ids = {doc.doc_id for doc in sample_documents}
        
        for chunk in sample_chunks:
            assert chunk.doc_id in doc_ids, \
                f"Chunk {chunk.chunk_id} references missing doc {chunk.doc_id}"
    
    def test_multi_chunks_structure(self, multi_chunks):
        """Multi-chunk fixture creates 2 chunks per document."""
        assert len(multi_chunks) == 16  # 8 docs * 2 chunks each
        
        # Check we have both _0 and _1 for each doc
        chunk_ids = [c.chunk_id for c in multi_chunks]
        assert "doc_001_0" in chunk_ids
        assert "doc_001_1" in chunk_ids