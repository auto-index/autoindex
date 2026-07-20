#!/bin/bash
#SBATCH --output=run_splits_logs/baseline_%x_%j.out
#SBATCH --error=run_splits_logs/baseline_%x_%j.err
#SBATCH --partition=cpu

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

SPLIT="${SPLIT:?SPLIT env var required}"

if [ ! -d "data/${SPLIT}" ]; then
    echo "[runtime] downloading dataset: ${SPLIT}"
    uv run python -m src.evaluation.scripts.get_data --split "${SPLIT}"
fi

START_EPOCH=$(date +%s)
echo "[runtime] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) split=${SPLIT} agent=baseline"

uv run python src/evaluation/scripts/test_preprocessing_split.py \
    --agent baseline \
    --split "${SPLIT}"
EXIT_CODE=$?

END_EPOCH=$(date +%s)
echo "[runtime] end=$(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed_seconds=$((END_EPOCH - START_EPOCH)) exit_code=$EXIT_CODE"
exit $EXIT_CODE
