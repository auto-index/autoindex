#!/bin/bash

# Always operate from the repository root (slurm jobs cd back via $SLURM_SUBMIT_DIR).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Dataset specifications: name|docs|queries|mem|time|cpus
# Resource notes (after first ablation pass):
#   - paper_retrieval bumped to 96G/12h: agent_noinput timed out at 6h.
#   - set_op / clinical_trial / tip_of_the_tongue bumped to 128G/24h: agent_noinput
#     timed out at 12h (BM25 reindex per hypothesis × 1M docs is the slow path).
#   - legal_qa bumped to 192G/24h: agent_noinput OOM'd at 96G (6,753 queries
#     materialised on top of the multi-hypothesis index footprint).
DATASET_SPECS=(
    "theorem_retrieval|23,839|69|32G|02:00:00|4"
    "stack_exchange|40,956|107|32G|02:00:00|4"
    "code_retrieval|232,444|3,665|64G|06:00:00|8"
    "paper_retrieval|363,133|72|96G|12:00:00|8"
    "set_operation_entity_retrieval|651,704|423|128G|24:00:00|12"
    "clinical_trial|914,628|113|128G|24:00:00|12"
    "tip_of_the_tongue|1,083,337|135|128G|24:00:00|12"
    "legal_qa|1,182,626|6,753|192G|24:00:00|12"
)

echo "=========================================="
echo "   AutoIndex - Dataset Runner"
echo "=========================================="
echo ""
echo "Available Models:"
echo "[1] sonnet  (openai/claude-sonnet-4-6 via UMass keymaker — recommended)"
echo "[2] haiku"
echo "[3] gpt4o"
echo "[4] qwen3coder  (openai/qwen/qwen3-coder via OpenRouter — needs OPENROUTER_API_KEY)"
echo ""
read -p "Select model (1-4): " model_selection

API_BASE=""
case $model_selection in
    1)
        MODEL="openai/claude-sonnet-4-6"
        API_BASE="https://thekeymaker.umass.edu/"
        ;;
    2)
        MODEL="haiku"
        ;;
    3)
        MODEL="gpt4o"
        ;;
    4)
        MODEL="openai/qwen/qwen3-coder"
        API_BASE="https://openrouter.ai/api/v1"
        if [ -z "$OPENROUTER_API_KEY" ]; then
            echo "ERROR: OPENROUTER_API_KEY env var not set. Run: export OPENROUTER_API_KEY=<your-key>"
            exit 1
        fi
        ;;
    *)
        echo "Invalid model selection. Exiting."
        exit 1
        ;;
esac

# Filename-safe short tag derived from MODEL (strips provider prefix, replaces dots).
# Used in slurm filenames + job names; the full MODEL is still passed to --model.
MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG//./-}"

echo ""
echo "Selected model: $MODEL"
echo ""
echo "Available Conditions:"
echo "[1] agent              - loop + analysis, no history, no contrastive"
echo "[2] agent_history      - loop + analysis + history of past hypotheses"
echo "[3] agent_contrastive  - loop + analysis + history + per-query contrastive table"
echo "[4] agent_noinput      - loop only, NO analysis, NO feedback, NO numbers"
echo ""
read -p "Select condition (1-4): " condition_selection

case $condition_selection in
    1) CONDITION="agent" ;;
    2) CONDITION="agent_history" ;;
    3) CONDITION="agent_contrastive" ;;
    4) CONDITION="agent_noinput" ;;
    *)
        echo "Invalid condition selection. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Selected condition: $CONDITION"
echo ""
echo "Available Datasets:"
echo ""
printf "%-5s %-35s %-12s %-10s %-8s %-10s %-6s\n" "NUM" "DATASET" "DOCS" "QUERIES" "MEMORY" "TIME" "CPUS"
echo "---------------------------------------------------------------------------------------------------"

i=1
for spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r name docs queries mem time cpus <<< "$spec"
    printf "%-5s %-35s %-12s %-10s %-8s %-10s %-6s\n" "[$i]" "$name" "$docs" "$queries" "$mem" "$time" "$cpus"
    ((i++))
done

echo ""
echo "Special options:"
echo "[9] Run ALL datasets"
echo "[0] Exit"
echo ""
read -p "Select dataset(s) to run (e.g., 1 3 5 or 9 for all): " selection

if [[ "$selection" == "0" ]]; then
    echo "Exiting."
    exit 0
fi

# Create ablation_slurm directory
mkdir -p ablation_slurm

# Function to submit a job
submit_job() {
    local spec=$1
    local model=$2
    local condition=$3
    local api_base=$4
    local model_tag=$5
    IFS='|' read -r name docs queries mem time cpus <<< "$spec"

    echo ""
    echo "Submitting: $name (model: $model, condition: $condition)"
    echo "  Documents: $docs | Queries: $queries"
    echo "  Resources: ${mem} RAM, ${cpus} CPUs, ${time}"

    cat > ablation_slurm/run_${name}_${model_tag}_${condition}.slurm << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=llm_${name}_${model_tag}_${condition}
#SBATCH --output=ablation_slurm/${name}_${model_tag}_${condition}_%j.out
#SBATCH --error=ablation_slurm/${name}_${model_tag}_${condition}_%j.err
#SBATCH --time=${time}
#SBATCH --mem=${mem}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --partition=cpu

# Go to the directory where the job was submitted from
cd \$SLURM_SUBMIT_DIR

# Defensive: kill any orphan BM25 server from a previous job on this node
pkill -u \$USER -f 'bm25_server' 2>/dev/null || true
sleep 2

export PATH="$HOME/.local/bin:$PATH"

# Download data if any required file is missing — checking the directory alone
# is unsafe: a partial / interrupted previous download leaves the dir present
# but documents.jsonl absent, and the agent crashes on _load_data.
if [ ! -f "data/${name}/documents.jsonl" ] || [ ! -f "data/${name}/validation_queries.jsonl" ] || [ ! -f "data/${name}/evaluation_queries.jsonl" ]; then
    echo "[runtime] Data missing for ${name}, downloading ..."
    uv run python -m src.evaluation.scripts.get_data --split ${name} || {
        echo "[runtime] ERROR: get_data failed for ${name}"
        exit 2
    }
fi

# Snapshot existing result files so we can identify the one this run produces
RESULTS_BEFORE=\$(mktemp)
ls -1 results/*/*.json 2>/dev/null | sort > "\$RESULTS_BEFORE"

# Run the agent with specified model — capture wall time at the bash layer
START_EPOCH=\$(date +%s)
echo "[runtime] start=\$(date -u +%Y-%m-%dT%H:%M:%SZ) split=${name} model=${model} condition=${condition} api_base=${api_base:-<native>}"
uv run python main.py --agent analysis_code_agent --loops 5 --condition ${condition} --split ${name} --model ${model} ${api_base:+--api-base "${api_base}"}
EXIT_CODE=\$?
END_EPOCH=\$(date +%s)
ELAPSED=\$((END_EPOCH - START_EPOCH))
echo "[runtime] end=\$(date -u +%Y-%m-%dT%H:%M:%SZ) split=${name} elapsed_seconds=\$ELAPSED exit_code=\$EXIT_CODE"

# Identify the result JSON this run produced (newest file not in pre-run snapshot)
RESULTS_AFTER=\$(mktemp)
ls -1 results/*/*.json 2>/dev/null | sort > "\$RESULTS_AFTER"
NEW_RESULT=\$(comm -13 "\$RESULTS_BEFORE" "\$RESULTS_AFTER" | tail -n 1)
rm -f "\$RESULTS_BEFORE" "\$RESULTS_AFTER"

# Extract latency block (wall time + token counts, total + per-agent) from result JSON
if [ -n "\$NEW_RESULT" ] && [ -f "\$NEW_RESULT" ]; then
    echo "[runtime] result file: \$NEW_RESULT"
    uv run python - <<PYEOF
import json, sys
with open("\$NEW_RESULT") as f:
    d = json.load(f)
lat = d.get("latency") or {}
print(f"[runtime] llm_total: wall={lat.get('total_wall_time_seconds', 0)}s  "
      f"calls={lat.get('llm_calls', 0)}  "
      f"prompt_tokens={lat.get('prompt_tokens', 0)}  "
      f"completion_tokens={lat.get('completion_tokens', 0)}  "
      f"total_tokens={lat.get('total_tokens', 0)}")
for agent_name, vals in (lat.get("by_agent") or {}).items():
    print(f"[runtime] llm_by_agent.{agent_name}: "
          f"calls={vals.get('calls', 0)}  "
          f"wall={vals.get('wall_time', 0):.1f}s  "
          f"prompt_tokens={vals.get('prompt_tokens', 0)}  "
          f"completion_tokens={vals.get('completion_tokens', 0)}")
PYEOF
else
    echo "[runtime] WARNING: no new result JSON found — token counts unavailable"
fi

echo "Job completed at \$(date)"
SLURM_EOF

    sbatch ablation_slurm/run_${name}_${model_tag}_${condition}.slurm
    echo "  ✓ Submitted as ablation_slurm/run_${name}_${model_tag}_${condition}.slurm"
}

# Process selection
if [[ "$selection" == "9" ]]; then
    echo ""
    echo "Submitting ALL datasets with model: $MODEL, condition: $CONDITION ..."
    for spec in "${DATASET_SPECS[@]}"; do
        submit_job "$spec" "$MODEL" "$CONDITION" "$API_BASE" "$MODEL_TAG"
        sleep 0.5
    done
else
    for num in $selection; do
        if [[ $num -ge 1 && $num -le ${#DATASET_SPECS[@]} ]]; then
            idx=$((num-1))
            submit_job "${DATASET_SPECS[$idx]}" "$MODEL" "$CONDITION" "$API_BASE" "$MODEL_TAG"
            sleep 0.5
        else
            echo "Invalid selection: $num"
        fi
    done
fi

echo ""
echo "=========================================="
echo "All jobs submitted with model: $MODEL, condition: $CONDITION"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER'"
echo ""
echo "Check logs:"
echo "  ls -lth ablation_slurm/"
echo "  tail -f ablation_slurm/<dataset>_${MODEL_TAG}_${CONDITION}_*.out"
echo "=========================================="