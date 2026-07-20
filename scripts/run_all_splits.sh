#!/usr/bin/env bash
# run_all_splits.sh — Run agent_history over all CRUMB splits sequentially.
#
# Usage:
#   bash run_all_splits.sh                                              # default model from config.yaml
#   bash run_all_splits.sh --model gemini/gemini-2.5-pro               # Gemini direct
#   bash run_all_splits.sh --model openai/gpt-4o-mini --api-base https://thekeymaker.umass.edu/
#
# Results are appended to results/<model>/ as each split finishes.
# To run only specific splits, edit SPLITS below.

set -euo pipefail

# Always operate from the repository root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Splits to run ────────────────────────────────────────────────────────────
SPLITS=(
    tip_of_the_tongue
    paper_retrieval
    clinical_trial
    legal_qa
    code_retrieval
    set_operation_entity_retrieval
    theorem_retrieval
    stack_exchange
)

# ── Parse flags ───────────────────────────────────────────────────────────────
MODEL_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_ARGS="$MODEL_ARGS --model $2"
            shift 2
            ;;
        --api-base)
            MODEL_ARGS="$MODEL_ARGS --api-base $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── Run ───────────────────────────────────────────────────────────────────────
FAILED=()

for SPLIT in "${SPLITS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  SPLIT: ${SPLIT}"
    echo "=============================================="

    # Check data is downloaded
    if [ ! -f "data/${SPLIT}/documents.jsonl" ] || [ ! -f "data/${SPLIT}/validation_queries.jsonl" ] || [ ! -f "data/${SPLIT}/evaluation_queries.jsonl" ]; then
        echo "  SKIP — data not found. Download with:"
        echo "    uv run python src/evaluation/scripts/get_data.py --split ${SPLIT}"
        FAILED+=("${SPLIT} (no data)")
        continue
    fi

    bash scripts/run_experiments.sh --split "${SPLIT}" $MODEL_ARGS || {
        echo "  ERROR: ${SPLIT} failed"
        FAILED+=("${SPLIT} (error)")
    }
done

echo ""
echo "=============================================="
echo "  ALL SPLITS COMPLETE"
echo "  Results in: results/"
echo "=============================================="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Skipped/failed: ${FAILED[*]}"
fi
ls -lt results/**/*.json results/*.json 2>/dev/null | head -20
