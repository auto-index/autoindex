#!/bin/bash
#SBATCH --output=run_splits_logs/abl_%x_%j.out
#SBATCH --error=run_splits_logs/abl_%x_%j.err
#SBATCH --partition=cpu

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

SPLIT="${SPLIT:?SPLIT required}"
AGENT="${AGENT:-analysis_code_agent}"
CONDITION="${CONDITION:-agent_history}"
MODEL="${MODEL:-openai/claude-sonnet-4-6}"

START_EPOCH=$(date +%s)
echo "[runtime] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) split=${SPLIT} agent=${AGENT} cond=${CONDITION} model=${MODEL}"

if [ "$AGENT" = "one_shot" ]; then
    uv run python -u main.py --agent one_shot --split "${SPLIT}" --model "${MODEL}"
else
    uv run python -u main.py --agent analysis_code_agent --loops 3 \
        --split "${SPLIT}" --condition "${CONDITION}" --model "${MODEL}"
fi
EXIT_CODE=$?

END_EPOCH=$(date +%s)
echo "[runtime] end=$(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed_seconds=$((END_EPOCH - START_EPOCH)) exit_code=$EXIT_CODE"
exit $EXIT_CODE
