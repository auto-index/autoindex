# Reproducing the paper's experiments

This folder documents how the experiments in *AutoIndex: Learning Representation Programs for Retrieval* were run. All entry points live in [`scripts/`](../scripts); everything below is invoked from the repository root.

## 1. Environment

```bash
uv sync
```

API keys go in `.env` at the repository root (see the main README). The paper's two Code Agent backbones were:

| Backbone | `--model` | `--api-base` | Seeds |
|---|---|---|---|
| qwen3-coder | `openai/qwen/qwen3-coder` | `https://openrouter.ai/api/v1` (needs `OPENROUTER_API_KEY`) | 3 |
| Claude Sonnet 4.6 | `openai/claude-sonnet-4-6` | LiteLLM-compatible proxy (we used a university-hosted one; needs `LITELLM_API_KEY`) | 2 |

Defaults for models/thresholds/timeouts are in `src/agents/analysis_code_agent/config.yaml`; CLI flags override them.

## 2. Data and query splits

Each CRUMB split is streamed from HuggingFace and cached under `data/<split>/` as `documents.jsonl`, `validation_queries.jsonl`, and `evaluation_queries.jsonl`:

```bash
uv run python -m src.evaluation.scripts.get_data --split tip_of_the_tongue   # one split
uv run python -m src.evaluation.scripts.get_data --all                       # everything (~4M docs)
```

The validation/evaluation partition (1:2 ratio, Appendix A.5 of the paper) is pinned by [`query_splits.json`](../query_splits.json) and applied during download. If you edit that file, re-apply it to cached data without re-downloading:

```bash
uv run python -m src.evaluation.scripts.repartition_splits
```

## 3. Baselines (BM25 full-document)

One chunk per document, no transformation (`src/agents/baseline/preprocess.py`):

```bash
bash scripts/run_baselines.sh                 # all splits, sequentially
# or a single split:
uv run python src/evaluation/scripts/test_preprocessing_split.py --agent baseline --split theorem_retrieval
```

On a Slurm cluster: `bash scripts/submit_all_baselines.sh` (uses `scripts/unity_baseline.slurm`).

The passage-corpus baseline numbers in the paper come from CRUMB's released passage corpus and evaluation protocol, not from a script in this repository.

## 4. Main results (Tables 1, 2, and 4)

Five optimization iterations, N=4 candidate programs per iteration, acceptance threshold ΔJ ≥ 1e-5 on validation Recall@100, best validation checkpoint scored once on held-out queries. One run = one seed; repeat per backbone for the seed counts above.

**Local / sequential** (small splits are fine on a laptop; the large ones need tens to hundreds of GB of RAM — see the resource table in `scripts/run_datasets.sh`):

```bash
bash scripts/run_all_splits.sh --model openai/qwen/qwen3-coder --api-base https://openrouter.ai/api/v1
# or one split:
bash scripts/run_experiments.sh --split theorem_retrieval --model openai/qwen/qwen3-coder --api-base https://openrouter.ai/api/v1
```

**Slurm** (what we used for the full corpora — interactive picker for model, condition, and splits, with per-split memory/time requests):

```bash
bash scripts/run_datasets.sh
```

Both paths ultimately run:

```bash
uv run python main.py --agent analysis_code_agent --loops 5 --condition <condition> \
    --split <split> --model <model> [--api-base <url>]
```

## 5. Ablations (Table 3)

The `--condition` flag controls which inputs the agents receive:

| Paper condition | `--condition` | What it does |
|---|---|---|
| Full AutoIndex (search history enabled) | `agent_history` | Analysis Agent + Code Agent + search history |
| w/o history | `agent` | Analysis Agent, but the Code Agent never sees past programs/outcomes |
| w/o analysis | `agent_noinput` | No Analysis Agent; Code Agent gets aggregate metrics only |
| 1 iter. | `agent_history` with `--loops 1` | Same pipeline, single iteration |

(`agent_contrastive` / `agent_contrastive_no_history` additionally give the Code Agent a per-query "what changed" table on top of history; these conditions exist in the code but are not part of the paper's ablation grid.)

Slurm helper for ablation conditions: `scripts/unity_ablation.slurm`, e.g.

```bash
sbatch --job-name=abl_tot --time=24:00:00 --mem=128G --cpus-per-task=12 \
    --export=ALL,SPLIT=tip_of_the_tongue,CONDITION=agent,MODEL=openai/qwen/qwen3-coder \
    scripts/unity_ablation.slurm
```

The one-shot program-generation baseline (single LLM call, no loop, no feedback):

```bash
uv run python main.py --agent one_shot --split <split> --model <model> [--api-base <url>]
```

## 6. Results and tables

Every run writes `results/<model>/<condition>_<timestamp>.json` containing per-iteration validation/held-out Recall@100 and nDCG@10, adopted program code, per-query outcomes, and per-agent token/latency accounting (Table 5). Iteration-level artifacts (analysis summaries, candidate programs, hypothesis scores) are kept under `ablation_experiments/<run>/`.

Aggregate into tables:

```bash
uv run python -m src.evaluation.scripts.generate_summary_table      # summary across runs
uv run python src/evaluation/scripts/generate_ablation_tables.py    # ablation tables
uv run python -m src.evaluation.scripts.aggregate_results           # cross-split comparison
```

Per-run iteration-dynamics plots (Figure 2 style):

```bash
uv run python src/agents/analysis_code_agent/plot_experiment.py --experiment-dir ablation_experiments/<run>
```

## Notes

- BM25 is `bm25s` v0.2.14, Lucene scoring, `k1=1.5, b=0.75`, lowercased regex word tokens with English stopword removal and no stemming; chunk→document aggregation is MaxP over 10,000 retrieved chunks per query. None of this is configurable from the agent's side — the retriever is fixed by design.
- Candidate programs run under a 15-minute execution timeout and must pass syntax validation (`config.yaml`).
- Full query-ID splits are also archived on OSF (see Appendix A.5 of the paper).
