# CodeAgent Test Documentation

## End-to-End Flow

Before reading the tests it helps to understand what `CodeAgent` actually does in one agent loop iteration. The diagram below shows every major step, the data that flows between them, and which tests exercise each step.

```
 INPUTS
 ──────
  analysis_summary  ← produced by the analysis agent (bash investigation output)
  current_code      ← the current preprocess.py on disk
  past_hypotheses   ← list of dicts from all previous iterations (empty on loop 1)
  documents         ← List[Document] loaded from corpus
  queries           ← List[EvalQuery] for evaluation
  BM25Client        ← HTTP client connected to the live BM25 server

        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: Assign taxonomy categories                                 │
│  _classify(description) + TAXONOMY_CATEGORIES                       │
│                                                                     │
│  Looks at past_hypotheses, classifies each by keyword matching into │
│  VOCABULARY / STRUCTURE / CONTEXT / QUERY-BRIDGING.                 │
│  Unexplored categories are put first so each iteration tries a      │
│  different strategy instead of repeating the same approach.         │
│                                                                     │
│  ← tested by: TestClassify, TestTaxonomyAssignment                  │
└─────────────────────────────────────────────────────────────────────┘
        │  assigned_categories = ["VOCABULARY", "CONTEXT", ...]
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: Generate hypotheses  (LLM call)                            │
│  generate_hypotheses(analysis_summary, current_code, ...)           │
│                                                                     │
│  Builds a prompt with: analysis summary, current code, dataset      │
│  info, past results, and the required category assignments.         │
│  Calls the LLM → gets back a markdown text block with H1–H4 headers │
│  each containing rationale, query IDs, and a full python code block.│
│                                                                     │
│  Then parses that markdown → List[Hypothesis]                        │
│  _parse_hypotheses_blocks(text)                                     │
│                                                                     │
│  ← tested by: TestParseHypothesesBlocks                             │
└─────────────────────────────────────────────────────────────────────┘
        │  hypotheses: List[Hypothesis]
        │  each Hypothesis has: id, description, rationale, code, query_ids
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: Test each hypothesis  (BM25 eval)                          │
│  test_hypothesis(hypothesis, documents, queries, current_code, ...)  │
│                                                                     │
│  3a. _validate_code(code, documents)                                │
│      Quick exec + preprocess() on 20-doc sample. Catches syntax     │
│      errors, wrong doc_ids, empty returns before hitting the server. │
│                                                                     │
│  3b. Run preprocess() on all documents → List[Chunk]                │
│      Build temporary BM25 index "hyp_{id}" on server.               │
│      Eval hypothesis index vs "current" index on all queries.       │
│      delta_recall@100 = hypothesis_recall - current_recall          │
│      proven = True if delta_recall@100 >= 0.05                      │
│      Track which query IDs improved / regressed.                    │
│      Delete temporary index.                                        │
│                                                                     │
│  Returns HypothesisResult with: recall scores, delta, proven flag,  │
│  improved_query_ids, regressed_query_ids, error (if any)            │
│                                                                     │
│  ← tested by: TestValidateCode (step 3a), TestTestHypothesis (3b)   │
└─────────────────────────────────────────────────────────────────────┘
        │  proven_results: List[HypothesisResult]
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: Generate final code  (LLM call, if any proven)             │
│  generate_final_code(analysis_summary, proven_results, current_code) │
│                                                                     │
│  Passes all proven hypotheses (code + deltas) to the LLM and asks  │
│  it to synthesize one combined preprocess.py. Writes it to disk.    │
│  (Not unit-tested — depends on a live LLM)                          │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
 OUTPUT: updated preprocess.py on disk, HypothesisResult list for
         the next iteration's past_hypotheses prompt context
```

---

## Run Commands

```bash
# Unit tests only (no server needed, ~3s)
uv run pytest tests/ -m "not integration" -v

# Integration tests only (starts BM25 server on port 8766, ~7s)
uv run pytest tests/ -m integration -v

# Full suite
uv run pytest tests/ -v
```

**Result: 30/30 passing (26 unit + 4 integration)**

---

## Fixtures and Shared Setup

### `conftest.py`

| Name | Scope | Description |
|------|-------|-------------|
| `SAMPLE_DOCS` | module constant | 4 synthetic `Document` objects — two Wikipedia-style articles ("The Matrix" `11111`, "Inception" `22222`), each with a title stub (`:0`) and a content section (`:1`) |
| `SAMPLE_QUERIES` | module constant | 2 `EvalQuery` objects whose gold docs are the content sections (`11111:1`, `22222:1`) — the title stubs alone won't match, so preprocessing that enriches them improves recall |
| `bm25_server` | session fixture | Starts a BM25 server subprocess on port 8766; waits up to 10s for health check; tears down after the full test session |
| `bm25_client_with_current_index` | function fixture | Pre-deletes any stale `"current"` index, then builds a passthrough index from `SAMPLE_DOCS`; deletes it after each test |

### `test_code_agent.py`

| Name | Scope | Description |
|------|-------|-------------|
| `MINIMAL_CONFIG` | module constant | Minimal `CodeAgent` config dict — `recall_improvement_threshold=0.05`, `server_port=8766`, no real API key needed for unit tests |
| `agent` | function fixture | Returns a fresh `CodeAgent(config=MINIMAL_CONFIG)` per test |
| `_PASSTHROUGH_CODE` | module constant | Valid passthrough `preprocess.py` string: one chunk per doc, text unchanged, correct `doc_id` |
| `_WELL_FORMED_FOUR_HYPOTHESES` | module constant | LLM-style markdown string with four complete hypothesis blocks (H1–H4), each with header, rationale, query IDs, falsifying condition, and a `class Preprocessor` code block |

---

## Unit Tests (26 tests)

The tests are ordered to follow the pipeline: taxonomy classification → category assignment → hypothesis parsing → code validation → full eval loop.

---

### `TestClassify` — 8 tests  *(Step 1: taxonomy keyword matching)*

**What this tests:** `CodeAgent._classify(description)` is a `@staticmethod` that maps a hypothesis description string to one of four taxonomy categories using keyword matching. It is called on every past hypothesis to figure out which categories have already been explored. The priority order is: CONTEXT > VOCABULARY > QUERY-BRIDGING > STRUCTURE, with VOCABULARY as the default fallback.

**Why it matters:** If `_classify` mislabels hypotheses, the diversity enforcement assigns the wrong categories and the agent may repeat approaches it has already tried instead of exploring new ones.

| Test | Description | Input | Expected |
|------|-------------|-------|----------|
| `test_title_prepend_is_context` | "title" is a CONTEXT keyword | `"Prepend title to all chunks"` | `"CONTEXT"` |
| `test_synonym_expansion_is_vocabulary` | "synonym" and "abbreviat" are VOCABULARY keywords | `"Add synonym expansion and abbreviation mapping"` | `"VOCABULARY"` |
| `test_sliding_window_is_structure` | "window" and "overlap" are STRUCTURE keywords | `"Sliding window chunking with overlap"` | `"STRUCTURE"` |
| `test_ngram_is_query_bridging` | "n-gram" is a QUERY-BRIDGING keyword | `"Extract n-gram phrases for BM25"` | `"QUERY-BRIDGING"` |
| `test_tfidf_is_query_bridging` | "tf-idf" is a QUERY-BRIDGING keyword | `"TF-IDF term reweighting"` | `"QUERY-BRIDGING"` |
| `test_context_takes_priority_over_vocabulary` | When both a CONTEXT and VOCABULARY keyword appear, CONTEXT wins because it's checked first | `"Prepend title and add synonym"` | `"CONTEXT"` |
| `test_unknown_description_defaults_to_vocabulary` | No keywords match → falls through to default | `"Something completely unrelated"` | `"VOCABULARY"` |
| `test_case_insensitive` | Keywords are matched case-insensitively | `"TITLE PREPEND STRATEGY"` / `"SYNONYM EXPANSION"` | `"CONTEXT"` / `"VOCABULARY"` |

---

### `TestTaxonomyAssignment` — 3 tests  *(Step 1: diversity enforcement)*

**What this tests:** The assignment logic inside `generate_hypotheses` that decides which taxonomy category each new hypothesis must target. It classifies past hypotheses with `_classify`, separates categories into "unexplored" and "explored", and returns `(unexplored + explored)[:n]` — so fresh categories always come first.

**Why it matters:** Without this, the LLM tends to keep generating chunking/structure variations. By forcing category assignments in the prompt, each iteration tries at least one genuinely different strategy.

The helper `_get_assigned_categories(past_hypotheses, n=4)` replicates the assignment logic directly without any LLM call, so these tests are fast and deterministic.

| Test | Description | Setup | Expected |
|------|-------------|-------|----------|
| `test_no_history_assigns_all_four_categories` | With no prior history, all four categories should be assigned | `past_hypotheses=[]` | Assigned set == `{"VOCABULARY", "STRUCTURE", "CONTEXT", "QUERY-BRIDGING"}` |
| `test_unexplored_categories_come_first` | After one CONTEXT hypothesis, the remaining three unexplored categories should fill slots 1–3 | One past hypothesis: `"Prepend title to chunks"` (→ CONTEXT) | First assigned category is NOT CONTEXT; CONTEXT still appears but is deprioritised to the end |
| `test_all_categories_tried_still_assigns_four` | Even when all categories have been tried, still return 4 assignments (cycles back through) | Four past hypotheses covering CONTEXT, VOCABULARY, STRUCTURE, QUERY-BRIDGING | Returns a list of exactly 4 |

---

### `TestParseHypothesesBlocks` — 10 tests  *(Step 2: parsing LLM output)*

**What this tests:** `CodeAgent._parse_hypotheses_blocks(text)` parses the raw markdown text returned by the LLM into a list of hypothesis dicts. It looks for `### H\d:` headers, extracts rationale/query IDs/falsifying condition from the text between headers, and associates each header with the `python` code block that follows it.

**Why it matters:** The LLM output is freeform text. If parsing fails or returns partial results, some hypotheses are silently dropped and never tested. The parser needs to be robust to missing fields while still rejecting hypotheses that have no code.

| Test | Description | Input | Expected |
|------|-------------|-------|----------|
| `test_parses_all_four_hypotheses` | Happy path — four well-formed hypothesis blocks | `_WELL_FORMED_FOUR_HYPOTHESES` | Returns list of exactly 4 dicts |
| `test_correct_ids` | IDs are extracted from the `### H1:` header | `_WELL_FORMED_FOUR_HYPOTHESES` | `["H1", "H2", "H3", "H4"]` in order |
| `test_correct_descriptions` | Description is the text after `### H1:` on the same line | `_WELL_FORMED_FOUR_HYPOTHESES` | "Title Prepend", "Synonym Expansion", "Sliding Window", "N-gram" |
| `test_extracts_rationale` | Rationale field is parsed from `Rationale: ...` line | `_WELL_FORMED_FOUR_HYPOTHESES` | H1 contains "Prepending the title"; H2 contains "vocabulary gap" |
| `test_extracts_query_ids` | Query IDs are parsed from `Query IDs: q1, q2` and split on commas | `_WELL_FORMED_FOUR_HYPOTHESES` | H1 has both `q_matrix` and `q_inception`; H2 has only `q_matrix` |
| `test_extracts_code_block_with_preprocessor_class` | The python code block between the header and next header is captured | `_WELL_FORMED_FOUR_HYPOTHESES` | Every hypothesis has `"class Preprocessor"` and `"def preprocess"` in its `code` field |
| `test_no_headers_returns_none` | No `### H\d:` markers at all → nothing to parse | Plain text | Returns `None` |
| `test_headers_without_code_blocks_returns_none` | Headers present but no ` ```python ``` ` blocks | Two headers, no code fences | Returns `None` |
| `test_partial_parse_returns_only_hypotheses_with_code` | H1 has a code block, H2 does not — only H1 should be returned | H1 + H2 text, only H1 has code | Returns list of length 1 with `id == "H1"` |
| `test_empty_string_returns_none` | Empty input | `""` | Returns `None` |

---

### `TestValidateCode` — 5 tests  *(Step 3a: pre-flight code check)*

**What this tests:** `CodeAgent._validate_code(code, documents)` catches broken hypothesis code *before* it hits the BM25 server. It execs the code string in a fresh namespace, calls `preprocess()` on a 20-doc sample, and checks that every returned chunk has a `doc_id` that matches one of the input documents.

**Why it matters:** Without this check, bad code would still build a partial index, produce misleading eval scores, or raise an unhandled exception mid-way. The validator short-circuits the expensive server round-trip for obviously wrong code.

| Test | Description | Code Under Test | Expected |
|------|-------------|-----------------|----------|
| `test_valid_passthrough_returns_none` | Correct code with proper `doc_id` passthrough should pass silently | `_PASSTHROUGH_CODE` | Returns `None` (no error) |
| `test_invalid_doc_id_returns_error` | A common LLM mistake: stripping `:section_idx` from the doc_id (e.g. `"11111:1"` → `"11111"`) produces chunks whose `doc_id` no longer matches any input document | `doc_id = d.doc_id.split(":")[0]` | Returns non-None string containing `"doc_id"` |
| `test_empty_return_returns_error` | `preprocess()` returns `[]` — no chunks produced, BM25 index would be empty | `return []` | Returns non-None string containing `"empty"` |
| `test_missing_preprocessor_class_returns_error` | File has no `class Preprocessor` — `load_preprocessor_from_code` can't find the class | `def some_function(): pass` | Returns non-None error string |
| `test_runtime_error_returns_error_string` | `preprocess()` raises an exception at runtime — error message should be captured and returned rather than propagating | `raise ValueError("intentional crash for testing")` | Returns non-None string containing `"intentional crash"` |

---

## Integration Tests (4 tests)

**Marker:** `@pytest.mark.integration`
**Requires:** Live BM25 server on port 8766 (started automatically by the `bm25_server` session fixture)
**Fixture dependency chain:** `bm25_client_with_current_index` → `bm25_server`

### `TestTestHypothesis` — 4 tests  *(Step 3b: full hypothesis eval loop)*

**What this tests:** The full `test_hypothesis()` method end-to-end: code validation → preprocessing → index build → BM25 eval → delta computation → proven flag → query-level improvement/regression tracking → index cleanup.

**Test corpus:** `SAMPLE_DOCS` (4 docs, 2 articles) + `SAMPLE_QUERIES` (2 queries targeting content sections). The passthrough `"current"` index only indexes each section as-is, so bare title stubs (`:0` docs) have very little vocabulary and miss the queries.

**Improving hypothesis (`_IMPROVING_CODE`):** Merges all sections of each Wikipedia article into a single concatenated text, then assigns that merged text to *every* section chunk of that article. This gives title stubs the full vocabulary of the content section, improving BM25 recall on the synthetic corpus.

| Test | Description | Hypothesis | Expected |
|------|-------------|------------|----------|
| `test_improving_hypothesis_is_proven` | A hypothesis that enriches document text should increase recall relative to the passthrough current index | `_IMPROVING_CODE` — section merging | `error is None`; `delta_recall_100 >= 0.0`; `hypothesis_recall_100 >= baseline_recall_100` |
| `test_passthrough_hypothesis_is_not_proven` | A hypothesis identical to the current index should produce zero delta — and should NOT cross the 0.05 threshold | `_PASSTHROUGH_CODE` | `error is None`; `delta_recall_100 ≈ 0.0` (within 1e-6); `proven is False` |
| `test_invalid_code_sets_error_not_proven` | Syntactically invalid code should be caught at the validation step, set `error`, and never reach the server | `"this is not valid python !!!"` | `error is not None`; `proven is False` |
| `test_improved_and_regressed_query_ids_populated` | A single query cannot both improve and regress in the same run — the two lists must be disjoint | `_IMPROVING_CODE` | `error is None`; `set(improved_query_ids) ∩ set(regressed_query_ids) == ∅` |
