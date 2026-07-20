# Testing Suite

## Overview

The test suite is organized to validate the project from the bottom up, starting with the data layer and building toward evaluation and agent behavior. The data layer tests ensure the pipeline can reliably fetch (mocked), transform, cache, and persist CRUMB-style documents and queries in the exact JSONL formats the rest of the system expects. Concretely, these tests check that raw HuggingFace dataset rows are converted into canonical documents.jsonl and queries.jsonl records (correct field mapping, qrels filtering, limits), that cache hits prevent unnecessary “downloads,” and that the pipeline behaves predictably end-to-end when run repeatedly.

Above that, the evaluation tests verify the static harness assumptions: schema correctness, preprocessor interface compliance, data quality rules, and cross-module integration so retrieval metrics are computed on valid inputs. Finally, the agent tests focus on agent-specific utilities (like BM25 client wrappers and evaluation helpers) to make sure the iterative loop has correct tooling and doesn’t break when the underlying evaluation outputs change. Together, this structure isolates failures: data issues are caught before they cascade into retrieval or agent logic, making debugging faster and results more trustworthy.

# 1. Data Layer Tests

## Test Structure

```
tests/
├── conftest.py                                    # Shared pytest fixtures and configuration
├── TESTS_README.md                                # Test documentation
│
├── data/                                          # Data Layer Tests
│   ├── __init__.py                                # Package initializer
│   ├── test_data_loader.py                        # Test data loading from various sources
│   ├── test_data_storage.py                       # Test data persistence and saving
│   ├── test_data_transformation.py                # Test data processing and transformations
│   ├── test_data_cache.py                         # Test caching mechanisms
│   └── test_data_pipeline.py                      # Test end-to-end data workflows
│
├── evaluation/                                    # Evaluation and validation tests
│   ├── test_base_preprocessor.py                  # Test base preprocessing functionality
│   ├── test_data_quality_validation.py            # Test data quality checks (7 tests)
│   ├── test_get_data.py                           # Test data retrieval functions
│   ├── test_pipeline_integration.py               # Test pipeline integration (2 tests, needs update)
│   └── test_schema.py                             # Test schema validation (3 tests)
│
└── agents/analysis_code_agent/                    # Agent-specific tests
    ├── test_bm25_client.py                        # Test BM25 client communication
    └── test_eval_utils.py                         # Test evaluation utilities
```

## Bulleted Description

tests/conftest.py

Central place for shared pytest fixtures (synthetic documents/queries/chunks, temp directories, mock datasets).
Ensures tests are fast, deterministic, and offline, without requiring CRUMB downloads.
tests/TESTS_README.md

Human-readable guide to the test suite: what each group covers, how to run them, expected behavior, and debugging tips.
Data layer tests (tests/data/)
These validate the “data plumbing” that feeds the evaluation harness: writing/reading JSONL, mapping HF dataset rows into the repo’s canonical formats, and ensuring cache + pipeline behavior is correct.

tests/data/test_data_loader.py

Tests data loading entrypoints (e.g., loading from HuggingFace via load_dataset, loading from local cached JSONL).
Verifies correct behavior for cache-hit vs cache-miss, limits (n_queries, limit), and failure cases—without making network calls (by mocking).
tests/data/test_data_storage.py

Tests persistence format and correctness of writing/reading JSONL artifacts.
Verifies the produced files are valid JSONL, contain required keys, and support round-trip integrity (write → read preserves fields).
tests/data/test_data_transformation.py

Tests the mapping/normalization logic inside the loader layer (in your case, get_data.py):
qrels fallback order (full_document_qrels → passage_qrels → empty)
filtering to label > 0
output shape for queries (query_id, query_content, relevant_doc_ids)
corpus row mapping (document_id → doc_id as string, document_content → text, metadata default)
Ensures you get consistent “canonical” records regardless of raw input shape.
tests/data/test_data_cache.py

Tests the caching contract:
correct cache directory naming (data/<split>/)
reading cached JSONL when it exists (no dataset calls)
writing cache files on cache miss
cache separation by split and limit (no collisions)
tests/data/test_data_pipeline.py

Tests the end-to-end data workflow (pipeline-style):
cache dir creation → queries load → corpus load → JSONL artifacts exist
second run uses cache (no “downloads”)
sanity checks like “relevant_doc_ids exist in docs” when the dataset supports it
documents the expected behavior when --limit is too small (relevance coverage may break)
Evaluation tests (tests/evaluation/)
These validate the static evaluation harness expectations and correctness around schemas, preprocessing interfaces, and evaluation integration.

tests/evaluation/test_base_preprocessor.py

Ensures preprocessors conform to the BasePreprocessor interface and expected invariants.
tests/evaluation/test_data_quality_validation.py

Checks data quality rules (IDs unique, required fields present, chunk/doc alignment, etc.) so eval results are meaningful.
tests/evaluation/test_get_data.py

Tests the get_data.py script/functions from the evaluation layer perspective (CLI behavior, retrieval logic, expected outputs).
tests/evaluation/test_pipeline_integration.py

Smoke/integration coverage across components (data → preprocess → index → eval), meant to catch wiring issues.
tests/evaluation/test_schema.py

Validates the schema dataclasses and fixture conformance (e.g., required keys, references valid).
Agent tests (tests/agents/analysis_code_agent/)
These validate agent-specific helper modules (not the static harness).

tests/agents/analysis_code_agent/test_bm25_client.py

Tests agent-side BM25 client behavior (request/response, ranking handling, error behavior).
tests/agents/analysis_code_agent/test_eval_utils.py

Tests agent-side evaluation utilities (metric calculations, formatting, result parsing, etc.).


## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/agents/ -v                              # All agent tests
pytest tests/evaluation/ -v                          # All evaluation tests

# Run specific test file
pytest tests/evaluation/test_data_quality_validation.py -v

# Run tests by pattern
pytest tests/ -k "agent" -v                          # All tests with "agent" in name
pytest tests/ -k "quality" -v                        # All quality-related tests

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Show detailed output (including print statements)
pytest tests/ -v -s

# Stop on first failure (fast feedback)
pytest tests/ -x

# Run specific test method
pytest tests/evaluation/test_data_quality_validation.py::TestDataQualityAcrossIterations::test_detects_missing_values -v
```