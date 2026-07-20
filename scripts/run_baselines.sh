#!/bin/bash
# Run the BM25 full-document baseline on all CRUMB splits.
# Invoke from anywhere: cd's to the repo root itself.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Log everything (stdout + stderr) to a timestamped file while still printing to console
mkdir -p logs
LOG_FILE="logs/run_baselines_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Datasets (full CRUMB corpus per split)
DATASETS=(
    "paper_retrieval"
    "legal_qa"
    "clinical_trial"
    "code_retrieval"
    "stack_exchange"
    "theorem_retrieval"
    "tip_of_the_tongue"
    "set_operation_entity_retrieval"
)

# Agents
AGENTS=(
    "baseline"
)

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  Running Baselines on All Datasets${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo "Datasets: ${#DATASETS[@]}"
echo "Agents: ${#AGENTS[@]}"
echo "Total runs: $((${#DATASETS[@]} * ${#AGENTS[@]}))"
echo ""

# Track results
SUCCESS=0
FAILED=0
START_TIME=$(date +%s)

# Run all combinations
for dataset in "${DATASETS[@]}"; do
    for agent in "${AGENTS[@]}"; do
        echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}▶ Running: ${agent} on ${dataset}${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        # Run evaluation
        uv run python src/evaluation/scripts/test_preprocessing_split.py \
            --agent ${agent} \
            --split ${dataset}
        
        # Check result
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Success: ${agent} on ${dataset}${NC}"
            ((SUCCESS++))
        else
            echo -e "${RED}✗ Failed: ${agent} on ${dataset}${NC}"
            ((FAILED++))
        fi
        
        # Brief pause between runs
        sleep 1
    done
done

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo -e "\n${BLUE}=================================================${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Successful: ${SUCCESS}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo -e "Duration: ${MINUTES}m ${SECONDS}s"
echo -e "\n${BLUE}Results saved to: results/{split}/{agent}_*.json${NC}"
echo ""

# Create summary report
uv run python - << 'PYTHON'
import json
from pathlib import Path

print("\n" + "="*90)
print("  Performance Summary")
print("="*90)

datasets = [
    "paper_retrieval",
    "legal_qa",
    "clinical_trial",
    "code_retrieval",
    "stack_exchange",
    "theorem_retrieval",
    "tip_of_the_tongue",
    "set_operation_entity_retrieval"
]

agents = ["baseline"]

print(f"\n{'Dataset':<45} {'Agent':<15} {'R@10':<10} {'R@100':<10} {'nDCG@10':<10}")
print("-"*90)

total_r10 = 0
total_r100 = 0
total_ndcg = 0
count = 0

for dataset in datasets:
    for agent in agents:
        results_dir = Path("results") / dataset
        if not results_dir.exists():
            continue
            
        pattern = f"{agent}_*_100.json"
        matches = list(results_dir.glob(pattern))
        
        if matches:
            try:
                with open(matches[0]) as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {})
                    r10 = metrics.get("recall_at_10", 0)
                    r100 = metrics.get("recall_at_100", 0) 
                    ndcg = metrics.get("ndcg_at_10", 0)
                    
                    print(f"{dataset:<45} {agent:<15} {r10:<10.4f} {r100:<10.4f} {ndcg:<10.4f}")
                    
                    total_r10 += r10
                    total_r100 += r100
                    total_ndcg += ndcg
                    count += 1
            except Exception as e:
                print(f"{dataset:<45} {agent:<15} ERROR: {e}")

if count > 0:
    print("-"*90)
    print(f"{'AVERAGE':<45} {'baseline':<15} {total_r10/count:<10.4f} {total_r100/count:<10.4f} {total_ndcg/count:<10.4f}")

print("\n")
PYTHON

echo -e "${GREEN}✓ Done! Baseline results ready.${NC}"