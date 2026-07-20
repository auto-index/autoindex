#!/bin/bash

# Always submit from the repository root so $SLURM_SUBMIT_DIR is the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# split|mem|time|cpus  — sized for BM25 indexing only, no LLM
SPECS=(
    "theorem_retrieval|16G|01:00:00|4"
    "stack_exchange|16G|01:00:00|4"
    "paper_retrieval|32G|02:00:00|4"
    "set_operation_entity_retrieval|48G|02:00:00|8"
    "clinical_trial|64G|03:00:00|8"
    "tip_of_the_tongue|64G|03:00:00|8"
    "code_retrieval|48G|02:00:00|8"
    "legal_qa|96G|04:00:00|12"
)

for spec in "${SPECS[@]}"; do
    IFS='|' read -r split mem time cpus <<< "$spec"
    echo "Submitting baseline for ${split} (mem=${mem} time=${time} cpus=${cpus})"
    sbatch \
        --job-name="bl_${split}" \
        --time="${time}" \
        --mem="${mem}" \
        --cpus-per-task="${cpus}" \
        --export=ALL,SPLIT="${split}" \
        scripts/unity_baseline.slurm
    sleep 0.3
done
