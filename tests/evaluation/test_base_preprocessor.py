"""Tests for BasePreprocessor abstract base class.

Validates that:
- BasePreprocessor cannot be instantiated directly
- Subclasses must implement preprocess()
- The preprocessor contract is enforced (chunks reference docs, unique IDs, etc.)
- Integration with fixture data works correctly
"""

import pytest
from abc import ABC
from schema import Document, Chunk
from base import BasePreprocessor


class TestBasePreprocessorAbstraction:
    """Tests for abstract base class behavior."""
    
    def test_cannot_instantiate_base_directly(self):
        """BasePreprocessor is abstract and cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            BasePreprocessor()
        
        assert "abstract" in str(exc_info.value).lower()
    
    def test_is_abstract_base_class(self):
        """BasePreprocessor inherits from ABC."""
        assert issubclass(BasePreprocessor, ABC)
    
    def test_has_abstract_preprocess_method(self):
        """preprocess() is marked as abstract."""
        assert hasattr(BasePreprocessor, 'preprocess')
        assert hasattr(BasePreprocessor.preprocess, '__isabstractmethod__')
        assert BasePreprocessor.preprocess.__isabstractmethod__ is True


class TestBasePreprocessorSubclassing:
    """Tests for subclassing behavior."""
    
    def test_subclass_without_preprocess_fails(self):
        """Subclass without preprocess() implementation cannot be instantiated."""
        class IncompletePreprocessor(BasePreprocessor):
            name = "incomplete"
            description = "Missing preprocess implementation"
            # Missing preprocess() method
        
        with pytest.raises(TypeError) as exc_info:
            IncompletePreprocessor()
        
        assert "abstract" in str(exc_info.value).lower()
        assert "preprocess" in str(exc_info.value).lower()
    
    def test_valid_subclass_can_be_instantiated(self):
        """Subclass with preprocess() can be instantiated."""
        class ValidPreprocessor(BasePreprocessor):
            name = "valid"
            description = "Valid implementation"
            
            def preprocess(self, docs):
                return [
                    Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text)
                    for d in docs
                ]
        
        preprocessor = ValidPreprocessor()
        assert preprocessor.name == "valid"
        assert preprocessor.description == "Valid implementation"
    
    def test_subclass_can_override_class_attributes(self):
        """Subclass can set custom name and description."""
        class CustomPreprocessor(BasePreprocessor):
            name = "my_custom_agent"
            description = "Custom chunking strategy"
            
            def preprocess(self, docs):
                return []
        
        preprocessor = CustomPreprocessor()
        assert preprocessor.name == "my_custom_agent"
        assert preprocessor.description == "Custom chunking strategy"
    
    def test_multiple_subclasses_independent(self):
        """Multiple subclasses can coexist with different implementations."""
        class PreprocessorA(BasePreprocessor):
            name = "agent_a"
            
            def preprocess(self, docs):
                return [Chunk(chunk_id="a", doc_id="d", text="A")]
        
        class PreprocessorB(BasePreprocessor):
            name = "agent_b"
            
            def preprocess(self, docs):
                return [Chunk(chunk_id="b", doc_id="d", text="B")]
        
        a = PreprocessorA()
        b = PreprocessorB()
        
        assert a.name != b.name
        assert a.preprocess([]) != b.preprocess([])


class TestPreprocessorContract:
    """Tests for the preprocessor contract using fixture data."""
    
    @pytest.fixture
    def simple_preprocessor(self):
        """A simple preprocessor that creates one chunk per document."""
        class SimplePreprocessor(BasePreprocessor):
            name = "simple"
            description = "One chunk per document"
            
            def preprocess(self, docs):
                return [
                    Chunk(
                        chunk_id=f"{doc.doc_id}_0",
                        doc_id=doc.doc_id,
                        text=doc.text,
                        metadata=doc.metadata
                    )
                    for doc in docs
                ]
        
        return SimplePreprocessor()
    
    def test_preprocess_returns_list_of_chunks(self, simple_preprocessor, sample_documents):
        """preprocess() returns a list of Chunk objects."""
        chunks = simple_preprocessor.preprocess(sample_documents)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_at_least_one_chunk_per_document(self, simple_preprocessor, sample_documents):
        """Contract: at least one chunk per document."""
        chunks = simple_preprocessor.preprocess(sample_documents)
        
        # Get unique doc_ids from chunks
        chunk_doc_ids = {chunk.doc_id for chunk in chunks}
        input_doc_ids = {doc.doc_id for doc in sample_documents}
        
        # Every input document should have at least one chunk
        assert chunk_doc_ids == input_doc_ids
    
    def test_chunk_doc_id_matches_source_document(self, simple_preprocessor, sample_documents):
        """Contract: chunk.doc_id must match source document's doc_id."""
        chunks = simple_preprocessor.preprocess(sample_documents)
        doc_ids = {doc.doc_id for doc in sample_documents}
        
        for chunk in chunks:
            assert chunk.doc_id in doc_ids, \
                f"Chunk {chunk.chunk_id} references unknown doc {chunk.doc_id}"
    
    def test_chunk_ids_globally_unique(self, simple_preprocessor, sample_documents):
        """Contract: chunk_id must be globally unique."""
        chunks = simple_preprocessor.preprocess(sample_documents)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        assert len(chunk_ids) == len(set(chunk_ids)), \
            f"Duplicate chunk IDs found: {[cid for cid in chunk_ids if chunk_ids.count(cid) > 1]}"
    
    def test_empty_document_list(self, simple_preprocessor):
        """Preprocessor handles empty document list."""
        chunks = simple_preprocessor.preprocess([])
        assert chunks == []
    
    def test_single_document(self, simple_preprocessor):
        """Preprocessor handles single document."""
        doc = Document(doc_id="single", text="Single document text")
        chunks = simple_preprocessor.preprocess([doc])
        
        assert len(chunks) >= 1
        assert chunks[0].doc_id == "single"


class TestMultiChunkPreprocessor:
    """Tests for preprocessors that create multiple chunks per document."""
    
    @pytest.fixture
    def multi_chunk_preprocessor(self):
        """Preprocessor that splits each document into 2 chunks."""
        class MultiChunkPreprocessor(BasePreprocessor):
            name = "multi_chunk"
            description = "Two chunks per document"
            
            def preprocess(self, docs):
                chunks = []
                for doc in docs:
                    # Split text roughly in half
                    mid = len(doc.text) // 2
                    chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_0",
                        doc_id=doc.doc_id,
                        text=doc.text[:mid]
                    ))
                    chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_1",
                        doc_id=doc.doc_id,
                        text=doc.text[mid:]
                    ))
                return chunks
        
        return MultiChunkPreprocessor()
    
    def test_multiple_chunks_per_document(self, multi_chunk_preprocessor, sample_documents):
        """Preprocessor can create multiple chunks per document."""
        chunks = multi_chunk_preprocessor.preprocess(sample_documents)
        
        # Should have 2 chunks per document
        assert len(chunks) == len(sample_documents) * 2
    
    def test_multi_chunk_ids_unique(self, multi_chunk_preprocessor, sample_documents):
        """All chunk IDs are unique even with multiple chunks per doc."""
        chunks = multi_chunk_preprocessor.preprocess(sample_documents)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        assert len(chunk_ids) == len(set(chunk_ids))
    
    def test_multi_chunk_all_reference_valid_docs(self, multi_chunk_preprocessor, sample_documents):
        """All chunks reference valid source documents."""
        chunks = multi_chunk_preprocessor.preprocess(sample_documents)
        doc_ids = {doc.doc_id for doc in sample_documents}
        
        for chunk in chunks:
            assert chunk.doc_id in doc_ids


class TestPreprocessorEdgeCases:
    """Tests for edge cases and error conditions."""
    
    @pytest.fixture
    def edge_case_preprocessor(self):
        """Preprocessor for testing edge cases."""
        class EdgeCasePreprocessor(BasePreprocessor):
            name = "edge_case"
            
            def preprocess(self, docs):
                return [
                    Chunk(
                        chunk_id=f"{doc.doc_id}_0",
                        doc_id=doc.doc_id,
                        text=doc.text
                    )
                    for doc in docs
                ]
        
        return EdgeCasePreprocessor()
    
    def test_document_with_empty_text(self, edge_case_preprocessor):
        """Preprocessor handles documents with empty text."""
        doc = Document(doc_id="empty", text="")
        chunks = edge_case_preprocessor.preprocess([doc])
        
        assert len(chunks) >= 1
        assert chunks[0].doc_id == "empty"
    
    def test_document_with_unicode(self, edge_case_preprocessor):
        """Preprocessor handles unicode text."""
        doc = Document(
            doc_id="unicode",
            text="Café 日本語 🎉"
        )
        chunks = edge_case_preprocessor.preprocess([doc])
        
        assert len(chunks) >= 1
        assert "Café" in chunks[0].text or "日本語" in chunks[0].text
    
    def test_document_with_long_text(self, edge_case_preprocessor):
        """Preprocessor handles very long documents."""
        long_text = "word " * 10000
        doc = Document(doc_id="long", text=long_text)
        chunks = edge_case_preprocessor.preprocess([doc])
        
        assert len(chunks) >= 1
        assert chunks[0].doc_id == "long"
    
    def test_many_documents(self, edge_case_preprocessor):
        """Preprocessor handles large document batches."""
        docs = [
            Document(doc_id=f"doc_{i}", text=f"Document {i} text")
            for i in range(1000)
        ]
        chunks = edge_case_preprocessor.preprocess(docs)
        
        assert len(chunks) >= 1000
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestPreprocessorWithFixtures:
    """Integration tests using conftest fixtures."""
    
    @pytest.fixture
    def baseline_preprocessor(self):
        """Baseline preprocessor matching baseline agent behavior."""
        class BaselinePreprocessor(BasePreprocessor):
            name = "baseline"
            description = "Passthrough - one chunk per document"
            
            def preprocess(self, docs):
                return [
                    Chunk(
                        chunk_id=f"{doc.doc_id}_0",
                        doc_id=doc.doc_id,
                        text=doc.text,
                        metadata=doc.metadata
                    )
                    for doc in docs
                ]
        
        return BaselinePreprocessor()
    
    def test_baseline_with_fixture_documents(self, baseline_preprocessor, sample_documents):
        """Baseline preprocessor works with fixture documents."""
        chunks = baseline_preprocessor.preprocess(sample_documents)
        
        assert len(chunks) == 8
        assert chunks[0].doc_id == "doc_001"
        assert "cat" in chunks[0].text.lower()
    
    def test_preprocessor_output_matches_sample_chunks(
        self, baseline_preprocessor, sample_documents, sample_chunks
    ):
        """Preprocessor output matches expected sample_chunks fixture."""
        chunks = baseline_preprocessor.preprocess(sample_documents)
        
        # Compare structure
        assert len(chunks) == len(sample_chunks)
        for chunk, expected in zip(chunks, sample_chunks):
            assert chunk.chunk_id == expected.chunk_id
            assert chunk.doc_id == expected.doc_id
            assert chunk.text == expected.text
    
    def test_preprocessor_preserves_document_content(
        self, baseline_preprocessor, sample_documents
    ):
        """Baseline preprocessor preserves original document text."""
        chunks = baseline_preprocessor.preprocess(sample_documents)
        
        for doc, chunk in zip(sample_documents, chunks):
            assert doc.text == chunk.text
            assert doc.doc_id == chunk.doc_id
