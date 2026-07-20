# scripts/generate_summary_table.py
import json
import pandas as pd
from pathlib import Path

def collect_all_experiments(results_root="results"):
    """Recursively find all config.json and final_results.json"""
    rows = []
    
    for config_file in Path(results_root).rglob("config.json"):
        exp_dir = config_file.parent
        final_results = exp_dir / "final_results.json"
        
        print(f"Found: {exp_dir}")
        
        try:
            with open(config_file) as f:
                config = json.load(f)
        except Exception as e:
            print(f"  ⚠ Could not read config: {e}")
            continue
        
        # Check if final_results exists
        if final_results.exists():
            try:
                with open(final_results) as f:
                    results = json.load(f)
            except Exception as e:
                print(f"  ⚠ Could not read final_results: {e}")
                results = {}
        else:
            print(f"  ⚠ No final_results.json found")
            results = {}
        
        rows.append({
            "model": config.get("model", "unknown"),
            "split": config.get("split", "unknown"),
            "loops": config.get("loops", "unknown"),
            "max_distractors": config.get("max_distractors", "unknown"),
            "n_datapoints": config.get("n_datapoints", config.get("max_distractors", "unknown")),
            "condition": config.get("condition", "unknown"),
            "baseline_recall_100": results.get("baseline_recall_100", None),
            "final_recall_100": results.get("final_recall_100", None),
            "improvement": results.get("improvement", None),
            "final_ndcg_10": results.get("final_ndcg_10", None),
            "timestamp": config.get("timestamp", "unknown"),
            "path": str(exp_dir.relative_to(results_root)),
        })
    
    if not rows:
        print(f"\n⚠ No experiments found in {results_root}")
        print(f"   Looking for: {Path(results_root).absolute()}/*/config.json")
        print(f"   Does this directory exist? {Path(results_root).exists()}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Only sort by columns that exist and aren't all 'unknown'
    sort_cols = []
    for col in ["model", "split", "timestamp"]:
        if col in df.columns and df[col].nunique() > 1:
            sort_cols.append(col)
    
    if sort_cols:
        df = df.sort_values(sort_cols)
    
    return df

if __name__ == "__main__":
    import sys
    results_root = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    print(f"Scanning {Path(results_root).absolute()} for experiments...\n")
    
    df = collect_all_experiments(results_root)
    
    if df.empty:
        print("\n❌ No experiments collected. Check the paths above.")
        sys.exit(1)
    
    output = Path(results_root) / "summary_all_experiments.csv"
    df.to_csv(output, index=False)
    print(f"\n✅ Saved {len(df)} experiments to {output}")
    print("\n" + df.to_string(index=False))