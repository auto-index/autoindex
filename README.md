# AutoIndex: Learning Representation Programs for Retrieval

AutoIndex is a framework for learning **representation programs**: executable transformations that map raw documents into the representations exposed to a retrieval system. Rather than tuning retrievers, rerankers, or a small set of preprocessing hyperparameters, AutoIndex searches over programs that preprocess documents before indexing. AutoIndex sets retrieval performance on a validation set as the objective, with the retriever (BM25) held fixed.

On [CRUMB](https://huggingface.co/datasets/jfkback/crumb), a benchmark of eight heterogeneous retrieval tasks, the learned programs improve recall over a static full-document BM25 baseline on all 8 tasks, with average gains of **+8.4% Recall@100** and **+8.3% nDCG@10** (largest gains: +30.5% Recall@100, +43.6% nDCG@10).

> **Paper:** *AutoIndex: Learning Representation Programs for Retrieval* — preprint, under review. See [Citation](#citation).

## How it works

```
                 ┌────────────────────────────────────────────────┐
                 │                AutoIndex loop                  │
                 │                                                │
 corpus ───────► │  representation program θ ──► build BM25 index │
 validation ───► │        ▲                          │            │
 queries         │        │                          ▼            │
                 │   Code Agent ◄── summary ── Analysis Agent     │
                 │   (N candidate       (diagnoses retrieval      │
                 │    programs)          failures with tools)     │
                 │        │                                       │
                 │        ▼                                       │
                 │  evaluate candidates on validation Recall@100; │
                 │  adopt improvements as the next incumbent      │
                 └────────────────────────────────────────────────┘
```

Each iteration:

1. **Index**: the current representation program (`preprocess.py`) is executed on the corpus and a BM25 index is built (`bm25s`, Lucene scoring, `k1=1.5, b=0.75`). Chunk scores are aggregated to documents with MaxP.
2. **Analyze**: the **Analysis Agent** diagnoses retrieval behavior under the current index using a read-only tool set (`bm25_retrieve`, `read_file`, `grep_search`) over a stratified slice of validation queries (regressions vs. the initial program, recall violations, and small-margin positives), and produces a structured failure summary grounded in concrete examples.
3. **Synthesize**: the **Code Agent** conditions on the summary and the search history (previously evaluated programs and their validation outcomes) and proposes N candidate programs.
4. **Select**: every candidate is executed, indexed, and scored on validation queries. Candidates that improve validation Recall@100 by at least a threshold are retained; if several survive, the LLM is asked to synthesize a combined program, adopted only if it beats the best individual candidate.

The best validation checkpoint is evaluated once on held-out queries. The learned program is applied at indexing time with no further LLM calls.

## Repository structure

```
autoindex/
├── main.py                       # CLI entry point: runs the AutoIndex loop
├── query_splits.json             # Validation/evaluation query ID partition (1:2) per split
├── scripts/                      # Experiment runners (see reproduce/README.md)
│   ├── run_all_splits.sh         #   AutoIndex over all CRUMB splits, sequentially
│   ├── run_experiments.sh        #   AutoIndex on one split
│   ├── run_baselines.sh          #   BM25 full-document baseline on all splits
│   ├── run_datasets.sh           #   Slurm launcher (interactive model/condition/split picker)
│   ├── unity_ablation.slurm      #   Slurm job template for ablation conditions
│   ├── unity_baseline.slurm      #   Slurm job template for the baseline
│   └── submit_all_baselines.sh   #   Submit baseline jobs for every split
├── reproduce/                    # Guide to reproducing the paper's experiments
├── src/
│   ├── evaluation/               # Static harness (fixed across all experiments)
│   │   ├── schema.py             #   Document / Chunk / EvalQuery dataclasses
│   │   ├── base.py               #   BasePreprocessor ABC — programs subclass this
│   │   └── scripts/
│   │       ├── get_data.py       #   Stream CRUMB splits from HuggingFace into data/
│   │       ├── build_index.py    #   BM25 index construction (bm25s)
│   │       ├── test_preprocessing_split.py  # Eval: Recall@100 + nDCG@10, MaxP aggregation
│   │       ├── repartition_splits.py        # Re-apply query_splits.json to cached data
│   │       ├── generate_summary_table.py    # Aggregate results into summary tables
│   │       ├── generate_ablation_tables.py  # Ablation result tables
│   │       └── aggregate_results.py         # Cross-split result comparison
│   └── agents/
│       ├── agent_runner.py       # Base class for iterative agents
│       ├── baseline/             # Full-document passthrough baseline (one chunk per doc)
│       └── analysis_code_agent/  # The AutoIndex implementation
│           ├── agent.py          #   Loop orchestrator (conditions, selection, results)
│           ├── analysis_agent.py #   Analysis Agent (tool-using diagnosis)
│           ├── code_agent.py     #   Code Agent (candidate program synthesis)
│           ├── one_shot_agent.py #   One-shot program-generation baseline
│           ├── analysis_tools/   #   bm25_retrieve / read_file / grep_search tools
│           ├── bm25_server.py    #   FastAPI server holding BM25 indexes in memory
│           ├── bm25_client.py    #   HTTP client for the server
│           ├── config.yaml       #   Models, thresholds, timeouts
│           └── context/          #   Agent system prompts + per-split corpus descriptions
└── tests/                        # Fast synthetic-fixture tests (no real corpus needed)
```

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/samonuall/autoindex.git
cd autoindex
uv sync
```

Provide LLM API keys in a `.env` file at the repository root (only the provider you use is required):

```bash
OPENROUTER_API_KEY=...   # for openai/<model> routed through OpenRouter (e.g. qwen3-coder)
GEMINI_API_KEY=...       # for gemini/<model> called natively
LITELLM_API_KEY=...      # for a LiteLLM-compatible proxy (used with --api-base)
```

## Quickstart

Download a CRUMB split (streamed from HuggingFace; cached under `data/<split>/`):

```bash
uv run python -m src.evaluation.scripts.get_data --split theorem_retrieval
```

Score the BM25 full-document baseline:

```bash
uv run python src/evaluation/scripts/test_preprocessing_split.py --agent baseline --split theorem_retrieval
```

Run the full AutoIndex loop (5 iterations, search history enabled):

```bash
uv run python main.py --agent analysis_code_agent --loops 5 --condition agent_history \
    --split theorem_retrieval --model openai/qwen/qwen3-coder --api-base https://openrouter.ai/api/v1
```

Results are written to `results/<model>/<condition>_<timestamp>.json`, including validation/held-out metrics per iteration, the adopted program code, and per-agent token usage. The learned program itself lives in `src/agents/analysis_code_agent/preprocess.py` (reset to the baseline at the start of every run).

### CRUMB splits

| Split | Abbrev. | Documents | Val queries | Eval queries |
|---|---|---|---|---|
| `clinical_trial` | CT | 914,628 | 41 | 84 |
| `code_retrieval` | CR | 232,444 | 1,255 | 2,510 |
| `legal_qa` | LQA | 1,182,626 | 2,284 | 4,569 |
| `paper_retrieval` | PR | 363,133 | 26 | 53 |
| `set_operation_entity_retrieval` | SOE | 651,704 | 156 | 314 |
| `stack_exchange` | SE | 40,956 | 39 | 79 |
| `theorem_retrieval` | TR | 23,839 | 25 | 51 |
| `tip_of_the_tongue` | TOT | 1,083,337 | 50 | 100 |

The validation/evaluation partition (1:2) is pinned by `query_splits.json` and applied automatically by `get_data.py`.

## Reproducing the paper

See [`reproduce/README.md`](reproduce/README.md) for the exact workflows behind the paper's tables — main results, ablations (search history, analysis agent, iteration count), baselines, and the Slurm setup used for the large splits.

## Tests

```bash
uv run pytest -m "not slow and not integration" -v   # fast tests, synthetic fixtures
uv run pytest -v                                     # everything
```

## Citation

```bibtex
@article{2026autoindex,
  title  = {AutoIndex: Learning Representation Programs for Retrieval},
  author = {O'Nuallain, Sam and Rajkumar, Nithya and Narayanasamy, Ramya and
            Jiang, Hanna and Chaudhari, Shreyas and Drozdov, Andrew},
  year   = {2026},
  note   = {Preprint.}
}
```

## License

MIT — see [LICENSE](LICENSE).
