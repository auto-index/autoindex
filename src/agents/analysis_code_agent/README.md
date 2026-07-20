# analysis_code_agent — the AutoIndex implementation

Two specialized agents iteratively improve a document representation program (`preprocess.py`)
for a fixed BM25 retriever:

1. **Analysis Agent** (`analysis_agent.py`) — diagnoses retrieval behavior under the current
   index and produces a structured failure summary.
2. **Code Agent** (`code_agent.py`) — conditions on that summary (plus the search history)
   and proposes N candidate programs, which are executed, indexed, and scored on validation
   queries before any of them is adopted.

## Files

```
analysis_code_agent/
├── agent.py             # Loop orchestrator: eval → analyze → synthesize → select
├── analysis_agent.py    # Tool-using Analysis Agent
├── analysis_tools/      # bm25_retrieve / read_file / grep_search tool implementations
├── code_agent.py        # Candidate generation, validation, testing, synthesis
├── one_shot_agent.py    # One-shot program-generation baseline (no loop)
├── eval_utils.py        # Validation-subset eval against the BM25 server
├── bm25_server.py       # FastAPI server holding named in-memory BM25 indexes
├── bm25_client.py       # HTTP client for the server
├── run_journal.py       # Search history: per-iteration hypotheses + outcomes
├── run_tracker.py       # Token / latency accounting per agent
├── plot_experiment.py   # Iteration-dynamics plots for a finished run
├── preprocess.py        # Current representation program (overwritten by the agent;
│                        #   reset to the baseline at the start of every run)
├── config.yaml          # Models, thresholds, timeouts, server settings
└── context/
    ├── ANALYSIS_SYSTEM.md        # Analysis Agent system prompt
    ├── CODE_SYSTEM.md            # Code Agent system prompt
    └── corpus_descriptions/      # Per-split corpus description injected into prompts
```

## Loop (per iteration)

1. Harness eval of the current `preprocess.py` (authoritative Recall@100 / nDCG@10).
2. Build the "current" index on the BM25 server and enrich validation queries with
   per-query hit/rank data.
3. **Analysis Agent** investigates a stratified slice of validation queries — regressions
   vs. the initial program, missed queries (with the top-ranked wrong documents), and
   worst-ranked successes — using three read-only tools:
   `bm25_retrieve(query, top_k)`, `read_file(file_path, max_chars, filter_id)`, and
   `grep_search(pattern, file_path, max_results)`. It must use tools before summarizing
   (up to `analysis_max_turns` turns).
4. **Code Agent** proposes up to `max_hypotheses` candidate programs in one call,
   conditioned on the analysis summary, the current code, and (when enabled) the run
   journal's search history.
5. Each candidate is syntax-validated, executed on the corpus under a timeout
   (`preprocess_timeout_seconds`), indexed on the BM25 server, and scored on validation
   queries.
6. The best candidate is adopted if it improves validation Recall@100. If multiple
   candidates cleared the improvement threshold, the Code Agent is asked to synthesize a
   combined program (`generate_final_code`), adopted only if it beats the best individual
   candidate on validation.
7. The globally best program across iterations is restored at the end and scored once on
   the held-out evaluation queries.

## Conditions

`main.py --condition` toggles the inputs (see `reproduce/README.md` for the mapping to the
paper's ablations):

- `agent_history` — analysis + search history (the paper's full pipeline)
- `agent` — analysis only, no history
- `agent_noinput` — no analysis; Code Agent sees aggregate metrics only
- `agent_contrastive` / `agent_contrastive_no_history` — additionally give the Code Agent a
  per-query "what changed" table (not part of the paper's ablation grid)

## Outputs

- `results/<model>/<condition>_<timestamp>.json` — final metrics, adopted code, per-query
  outcomes, token/latency stats
- `ablation_experiments/<run>/` — per-iteration analysis summaries, hypotheses, and journal
- `logs/` — full LLM conversations per iteration
