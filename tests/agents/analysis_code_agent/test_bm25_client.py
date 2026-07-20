"""Data validation tests for bm25_client.py.

Tests focus on:
- Request data formatting and validation
- Response data structure validation
- Error handling and retry logic
- Edge cases (empty data, malformed responses)
- Integration with Chunk and EvalQuery schemas
"""

import sys
import pathlib
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import httpx

# Add agent to path
sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "src" / "agents" / "analysis_code_agent"))

from bm25_client import BM25Client, _with_retry
from schema import Chunk, EvalQuery


class TestBM25ClientInitialization:
    """Test client initialization and configuration."""
    
    def test_client_default_initialization(self):
        """Client can be initialized with defaults."""
        client = BM25Client()
        
        assert client.base_url == "http://localhost:8765"
        assert client.timeout == 600.0
        assert client.max_retries == 3
    
    def test_client_custom_initialization(self):
        """Client can be initialized with custom settings."""
        client = BM25Client(
            base_url="http://custom:9999",
            timeout=60.0,
            max_retries=5
        )
        
        assert client.base_url == "http://custom:9999"
        assert client.timeout == 60.0
        assert client.max_retries == 5
    
    def test_client_base_url_validation(self):
        """Base URL is stored correctly."""
        client = BM25Client(base_url="http://example.com:8000")
        assert "example.com" in client.base_url
        assert "8000" in client.base_url


class TestBuildIndexDataValidation:
    """Test build_index request data validation."""
    
    @pytest.fixture
    def client(self):
        return BM25Client()
    
    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx.Client as a context manager."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        return mock_client
    
    def test_build_index_formats_chunks_correctly(self, client, sample_chunks, mock_httpx_client):
        """build_index converts Chunk objects to proper dict format."""
        with patch('httpx.Client', return_value=mock_httpx_client):
            client.build_index("test_index", sample_chunks)
            
            # Check the call was made
            assert mock_httpx_client.post.called
            call_args = mock_httpx_client.post.call_args
            
            # Validate endpoint
            assert call_args[0][0] == "/index/test_index/build"
            
            # Validate payload structure
            payload = call_args[1]['json']
            assert 'chunks' in payload
            assert 'persist' in payload
            assert isinstance(payload['chunks'], list)
            assert len(payload['chunks']) == len(sample_chunks)
    
    def test_build_index_chunk_fields_present(self, client, mock_httpx_client):
        """Each chunk in payload has required fields."""
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", text="text1", metadata={"k": "v"}),
            Chunk(chunk_id="c2", doc_id="d2", text="text2", metadata={}),
        ]
        
        with patch('httpx.Client', return_value=mock_httpx_client):
            client.build_index("idx", chunks)
            
            payload = mock_httpx_client.post.call_args[1]['json']
            
            for chunk_dict in payload['chunks']:
                assert 'chunk_id' in chunk_dict
                assert 'doc_id' in chunk_dict
                assert 'text' in chunk_dict
                assert 'metadata' in chunk_dict
    
    def test_build_index_preserves_chunk_data(self, client, mock_httpx_client):
        """Chunk data is preserved correctly in payload."""
        chunk = Chunk(
            chunk_id="test_chunk_123",
            doc_id="test_doc_456",
            text="Test text content",
            metadata={"key": "value", "number": 42}
        )
        
        with patch('httpx.Client', return_value=mock_httpx_client):
            client.build_index("idx", [chunk])
            
            payload = mock_httpx_client.post.call_args[1]['json']
            chunk_dict = payload['chunks'][0]
            
            assert chunk_dict['chunk_id'] == "test_chunk_123"
            assert chunk_dict['doc_id'] == "test_doc_456"
            assert chunk_dict['text'] == "Test text content"
            assert chunk_dict['metadata']['key'] == "value"
            assert chunk_dict['metadata']['number'] == 42
    
    def test_build_index_empty_chunks(self, client, mock_httpx_client):
        """build_index handles empty chunk list."""
        with patch('httpx.Client', return_value=mock_httpx_client):
            client.build_index("empty_idx", [])
            
            payload = mock_httpx_client.post.call_args[1]['json']
            assert payload['chunks'] == []
    
    def test_build_index_persist_flag(self, client, mock_httpx_client):
        """persist flag is passed correctly."""
        chunks = [Chunk(chunk_id="c1", doc_id="d1", text="t1")]
        
        with patch('httpx.Client', return_value=mock_httpx_client):
            # Default persist=False
            client.build_index("idx", chunks)
            payload = mock_httpx_client.post.call_args[1]['json']
            assert payload['persist'] is False
            
            # Explicit persist=True
            client.build_index("idx", chunks, persist=True)
            payload = mock_httpx_client.post.call_args[1]['json']
            assert payload['persist'] is True
    
    def test_build_index_unicode_text(self, client, mock_httpx_client):
        """build_index handles unicode in chunk text."""
        chunk = Chunk(
            chunk_id="unicode_chunk",
            doc_id="unicode_doc",
            text="Café 日本語 🎉",
            metadata={}
        )
        
        with patch('httpx.Client', return_value=mock_httpx_client):
            client.build_index("idx", [chunk])
            
            payload = mock_httpx_client.post.call_args[1]['json']
            assert "Café" in payload['chunks'][0]['text']
            assert "日本語" in payload['chunks'][0]['text']
            assert "🎉" in payload['chunks'][0]['text']


class TestRetrieveDataValidation:
    """Test retrieve response data validation."""
    
    @pytest.fixture
    def client(self):
        return BM25Client()
    
    @pytest.fixture
    def mock_retrieve_response(self):
        """Mock successful retrieve response."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "results": [
                {"doc_id": "doc1", "score": 1.5, "rank": 1},
                {"doc_id": "doc2", "score": 1.2, "rank": 2},
                {"doc_id": "doc3", "score": 0.9, "rank": 3},
            ]
        }
        return mock_response
    
    def test_retrieve_returns_list(self, client, mock_retrieve_response):
        """retrieve returns a list of result dicts."""
        mock_client = Mock()
        mock_client.post.return_value = mock_retrieve_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            results = client.retrieve("test_idx", "test query")
            
            assert isinstance(results, list)
            assert len(results) == 3
    
    def test_retrieve_result_structure(self, client, mock_retrieve_response):
        """Each result has expected fields."""
        mock_client = Mock()
        mock_client.post.return_value = mock_retrieve_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            results = client.retrieve("idx", "query")
            
            for result in results:
                assert 'doc_id' in result
                assert 'score' in result
                assert 'rank' in result
    
    def test_retrieve_query_formatting(self, client, mock_retrieve_response):
        """retrieve sends query in correct format."""
        mock_client = Mock()
        mock_client.post.return_value = mock_retrieve_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            client.retrieve("idx", "test query text", top_k=50)
            
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "/index/idx/retrieve"
            
            payload = call_args[1]['json']
            assert payload['query'] == "test query text"
            assert payload['top_k'] == 50
    
    def test_retrieve_empty_results(self, client):
        """retrieve handles empty results."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"results": []}
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            results = client.retrieve("idx", "query")
            assert results == []
    
    def test_retrieve_unicode_query(self, client, mock_retrieve_response):
        """retrieve handles unicode in query text."""
        mock_client = Mock()
        mock_client.post.return_value = mock_retrieve_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            client.retrieve("idx", "Café 日本語 🎉")
            
            payload = mock_client.post.call_args[1]['json']
            assert "Café" in payload['query']
            assert "日本語" in payload['query']


class TestBatchRetrieveDataValidation:
    """Test batch_retrieve request and response validation."""
    
    @pytest.fixture
    def client(self):
        return BM25Client()
    
    @pytest.fixture
    def sample_queries_for_batch(self):
        """Sample EvalQuery objects."""
        return [
            EvalQuery(query_id="q1", query_text="query text 1", relevant_doc_ids=["d1"]),
            EvalQuery(query_id="q2", query_text="query text 2", relevant_doc_ids=["d2"]),
            EvalQuery(query_id="q3", query_text="query text 3", relevant_doc_ids=["d3"]),
        ]
    
    @pytest.fixture
    def mock_batch_response(self):
        """Mock batch retrieve response."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "results": [
                {"query_id": "q1", "ranked_docs": [{"doc_id": "d1", "score": 1.0}]},
                {"query_id": "q2", "ranked_docs": [{"doc_id": "d2", "score": 0.9}]},
                {"query_id": "q3", "ranked_docs": []},
            ]
        }
        return mock_response
    
    def test_batch_retrieve_query_formatting(self, client, sample_queries_for_batch, mock_batch_response):
        """batch_retrieve formats queries correctly."""
        mock_client = Mock()
        mock_client.post.return_value = mock_batch_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            client.batch_retrieve("idx", sample_queries_for_batch, top_k=100)
            
            payload = mock_client.post.call_args[1]['json']
            
            assert 'queries' in payload
            assert 'top_k' in payload
            assert len(payload['queries']) == 3
    
    def test_batch_retrieve_query_fields(self, client, sample_queries_for_batch, mock_batch_response):
        """Each query in payload has query_id and query_text."""
        mock_client = Mock()
        mock_client.post.return_value = mock_batch_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            client.batch_retrieve("idx", sample_queries_for_batch)
            
            payload = mock_client.post.call_args[1]['json']
            
            for query_dict in payload['queries']:
                assert 'query_id' in query_dict
                assert 'query_text' in query_dict
                # relevant_doc_ids should NOT be in the request
                assert 'relevant_doc_ids' not in query_dict
    
    def test_batch_retrieve_preserves_query_data(self, client, mock_batch_response):
        """Query data is preserved in request."""
        queries = [
            EvalQuery(query_id="test_q1", query_text="test query text", relevant_doc_ids=[])
        ]
        
        mock_client = Mock()
        mock_client.post.return_value = mock_batch_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            client.batch_retrieve("idx", queries)
            
            payload = mock_client.post.call_args[1]['json']
            query_dict = payload['queries'][0]
            
            assert query_dict['query_id'] == "test_q1"
            assert query_dict['query_text'] == "test query text"
    
    def test_batch_retrieve_response_structure(self, client, sample_queries_for_batch, mock_batch_response):
        """Response has correct structure."""
        mock_client = Mock()
        mock_client.post.return_value = mock_batch_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            results = client.batch_retrieve("idx", sample_queries_for_batch)
            
            assert isinstance(results, list)
            assert len(results) == 3
            
            for result in results:
                assert 'query_id' in result
                assert 'ranked_docs' in result
                assert isinstance(result['ranked_docs'], list)
    
    def test_batch_retrieve_empty_queries(self, client):
        """batch_retrieve handles empty query list."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"results": []}
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            results = client.batch_retrieve("idx", [])
            
            assert results == []


class TestDeleteIndexValidation:
    """Test delete_index operation."""
    
    @pytest.fixture
    def client(self):
        return BM25Client()
    
    def test_delete_index_endpoint(self, client):
        """delete_index calls correct endpoint."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        
        mock_client = Mock()
        mock_client.delete.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            client.delete_index("test_idx")
            
            assert mock_client.delete.called
            call_args = mock_client.delete.call_args
            assert call_args[0][0] == "/index/test_idx"
    
    def test_delete_index_special_characters(self, client):
        """delete_index handles index names with special characters."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        
        mock_client = Mock()
        mock_client.delete.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            client.delete_index("hyp_H1_test")
            
            call_args = mock_client.delete.call_args
            assert "hyp_H1_test" in call_args[0][0]


class TestHealthCheckValidation:
    """Test health check endpoint."""
    
    @pytest.fixture
    def client(self):
        return BM25Client()
    
    def test_health_returns_true_on_200(self, client):
        """health returns True when server responds with 200."""
        mock_response = Mock()
        mock_response.status_code = 200
        
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            assert client.health() is True
    
    def test_health_returns_false_on_error(self, client):
        """health returns False on exception."""
        mock_client = Mock()
        mock_client.get.side_effect = httpx.ConnectError("Connection failed")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            assert client.health() is False
    
    def test_health_returns_false_on_non_200(self, client):
        """health returns False on non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500
        
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            assert client.health() is False


class TestRetryLogic:
    """Test retry decorator behavior."""
    
    def test_retry_on_request_error(self):
        """Function retries on httpx.RequestError."""
        call_count = 0
        
        @_with_retry(max_attempts=3, backoff=0.01)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.RequestError("Connection failed")
            return "success"
        
        result = failing_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhaustion_raises_error(self):
        """Function raises error after max retries."""
        @_with_retry(max_attempts=3, backoff=0.01)
        def always_fails():
            raise httpx.RequestError("Always fails")
        
        with pytest.raises(httpx.RequestError):
            always_fails()
    
    def test_no_retry_on_success(self):
        """Function doesn't retry on success."""
        call_count = 0
        
        @_with_retry(max_attempts=3, backoff=0.01)
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = succeeds()
        
        assert result == "success"
        assert call_count == 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def client(self):
        return BM25Client()
    
    def test_http_error_propagates(self, client):
        """HTTP errors are raised after retries."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=Mock(status_code=404)
        )
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                client.retrieve("idx", "query")
    
    def test_timeout_error_retries(self, client):
        """Timeout errors trigger retry logic."""
        call_count = 0
        
        def timeout_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("Timeout")
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"results": []}
            return mock_response
        
        mock_client = Mock()
        mock_client.post.side_effect = timeout_then_succeed
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        
        with patch('httpx.Client', return_value=mock_client):
            # Should succeed after retry
            results = client.retrieve("idx", "query")
            assert call_count == 2
