"""
Aggregate and compare all evaluation results across agents and splits.

Usage:
    python -m src.evaluation.scripts.aggregate_results
    python -m src.evaluation.scripts.aggregate_results --split paper_retrieval
    python -m src.evaluation.scripts.aggregate_results --agent baseline
    python -m src.evaluation.scripts.aggregate_results --export results.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

_PROJECT_ROOT = Path(__file__).parents[3]
RESULTS_DIR = _PROJECT_ROOT / "results"


def load_all_results(
    split_filter: Optional[str] = None,
    agent_filter: Optional[str] = None,
) -> List[Dict]:
    """Load all result JSON files, optionally filtered by split or agent."""
    if not RESULTS_DIR.exists():
        return []
    
    results = []
    
    for split_dir in RESULTS_DIR.iterdir():
        if not split_dir.is_dir():
            continue
        
        split_name = split_dir.name
        if split_filter and split_name != split_filter:
            continue
        
        for result_file in split_dir.glob("*.json"):
            filename = result_file.name
            
            # Skip non-JSON files
            if not filename.endswith(".json"):
                continue
            
            # Skip CRUMB format files (end with _crumb.jsonl but might be _crumb.json)
            if "_crumb" in filename:
                continue
            
            # Skip query results files
            if "_results" in filename:
                continue
            
            # Only load main summary files (should have pattern: agent_Ndocs_kK.json)
            # Skip if it doesn't have the expected pattern
            if not ("docs_k" in filename or "docs_" in filename):
                continue
            
            try:
                with result_file.open('r') as f:
                    data = json.load(f)
                
                agent_name = data.get('agent', 'unknown')
                if agent_filter and agent_name != agent_filter:
                    continue
                
                # Build flattened record
                record = {
                    'agent': agent_name,
                    'split': data.get('split', split_name),
                    'timestamp': data.get('timestamp', 'unknown'),
                    'n_docs': data.get('config', {}).get('n_docs', 0),
                    'n_queries': data.get('config', {}).get('n_queries', 0),
                    'chunks_per_doc': data.get('config', {}).get('chunks_per_doc', 0),
                }
                
                # Add quick metrics
                metrics = data.get('metrics', {})
                if metrics:
                    # Load the correct metric keys from the evaluation results
                    record['recall@100'] = metrics.get('recall_at_100', None)
                    record['recall@1000'] = metrics.get('recall_at_1000', None)
                    record['ndcg@10_quick'] = metrics.get('ndcg_at_10', None)
                else:
                    record['recall@100'] = None
                    record['recall@1000'] = None
                    record['ndcg@10_quick'] = None
                
                # Add CRUMB metrics if available
                crumb = data.get('crumb_metrics', {}) or {}
                if crumb:
                    record['nDCG@10'] = crumb.get('nDCG@10', None)
                    record['nDCG@5'] = crumb.get('nDCG@5', None)
                    # Use the correct CRUMB metric keys
                    record['R@100'] = crumb.get('R@100', None)
                    record['R@1000'] = crumb.get('R@1000', None)
                    record['P@10'] = crumb.get('P@10', None)
                    record['RR@10'] = crumb.get('RR@10', None)
                else:
                    record['nDCG@10'] = None
                    record['nDCG@5'] = None
                    record['R@100'] = None
                    record['R@1000'] = None
                    record['P@10'] = None
                    record['RR@10'] = None
                
                results.append(record)
                
            except Exception as e:
                print(f"⚠️  Error loading {result_file}: {e}", file=sys.stderr)
    
    return results


def print_comparison_table(results: List[Dict], sort_by: str = 'nDCG@10'):
    """Print results as a formatted comparison table."""
    if not results:
        print("No results found.")
        return
    
    # Sort by metric (descending, NaN last) then by agent
    def sort_key(r):
        # Try CRUMB metric first, then fall back to quick metric
        val = r.get(sort_by)
        if val is None and sort_by == 'nDCG@10':
            val = r.get('ndcg@10_quick')  # Fallback to quick metric
        if val is None:
            return (1, r['agent'])  # NaN last
        return (0, -val, r['agent'])
    
    results = sorted(results, key=sort_key)
    
    print("\n" + "="*140)
    print("EVALUATION RESULTS COMPARISON")
    print("="*140)
    print(f"\n{'Agent':<20} {'Split':<25} {'Docs':<10} {'Queries':<8} {'Chunks/Doc':<12} {'nDCG@10':<10} {'R@100':<10} {'R@1000':<10}")
    print("-"*140)
    
    # Track values for averaging
    sum_docs = 0
    sum_queries = 0
    sum_chunks_per_doc = 0
    sum_ndcg = 0
    sum_r100 = 0
    sum_r1000 = 0
    count_ndcg = 0
    count_r100 = 0
    count_r1000 = 0
    
    for r in results:
        # Try CRUMB metrics first, fall back to quick metrics
        ndcg = r.get('nDCG@10') or r.get('ndcg@10_quick')
        r100 = r.get('R@100') or r.get('recall@100')
        r1000 = r.get('R@1000') or r.get('recall@1000')
        
        ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "N/A"
        r100_str = f"{r100:.4f}" if r100 is not None else "N/A"
        r1000_str = f"{r1000:.4f}" if r1000 is not None else "N/A"
        
        # Try CRUMB metrics first, fall back to quick metrics
        ndcg = r.get('nDCG@10') or r.get('ndcg@10_quick')
        r100 = r.get('R@100') or r.get('recall@100')
        r1000 = r.get('R@1000') or r.get('recall@1000')

        # Mark each metric with asterisk if using fallback
        ndcg_suffix = "*" if (r.get('nDCG@10') is None and ndcg is not None) else ""
        r100_suffix = "*" if (r.get('R@100') is None and r100 is not None) else ""
        r1000_suffix = "*" if (r.get('R@1000') is None and r1000 is not None) else ""

        ndcg_str = f"{ndcg:.4f}{ndcg_suffix}" if ndcg is not None else "N/A"
        r100_str = f"{r100:.4f}{r100_suffix}" if r100 is not None else "N/A"
        r1000_str = f"{r1000:.4f}{r1000_suffix}" if r1000 is not None else "N/A"

        print(
            f"{r['agent']:<20} {r['split']:<25} {r['n_docs']:<10,} "
            f"{r['n_queries']:<8} {r['chunks_per_doc']:<12.2f} "
            f"{ndcg_str:<11} {r100_str:<11} {r1000_str:<11}"
        )
        
        # Accumulate for averages
        sum_docs += r['n_docs']
        sum_queries += r['n_queries']
        sum_chunks_per_doc += r['chunks_per_doc']
        
        if ndcg is not None:
            sum_ndcg += ndcg
            count_ndcg += 1
        if r100 is not None:
            sum_r100 += r100
            count_r100 += 1
        if r1000 is not None:
            sum_r1000 += r1000
            count_r1000 += 1
    
    # Print averages row
    print("-"*140)
    n = len(results)
    avg_docs = sum_docs / n if n > 0 else 0
    avg_queries = sum_queries / n if n > 0 else 0
    avg_chunks_per_doc = sum_chunks_per_doc / n if n > 0 else 0
    avg_ndcg = sum_ndcg / count_ndcg if count_ndcg > 0 else None
    avg_r100 = sum_r100 / count_r100 if count_r100 > 0 else None
    avg_r1000 = sum_r1000 / count_r1000 if count_r1000 > 0 else None
    
    avg_ndcg_str = f"{avg_ndcg:.4f}" if avg_ndcg is not None else "N/A"
    avg_r100_str = f"{avg_r100:.4f}" if avg_r100 is not None else "N/A"
    avg_r1000_str = f"{avg_r1000:.4f}" if avg_r1000 is not None else "N/A"
    
    print(
        f"{'AVERAGE':<20} {'':<25} {avg_docs:<10,.0f} "
        f"{avg_queries:<8.0f} {avg_chunks_per_doc:<12.2f} "
        f"{avg_ndcg_str:<10} {avg_r100_str:<10} {avg_r1000_str:<10}"
    )
    
    print("="*140)
    print(f"\nTotal runs: {len(results)}")
    print(f"Sorted by: {sort_by} (descending)")
    
    # Check if any used fallback metrics
    has_fallback = any(r.get('nDCG@10') is None and r.get('ndcg@10_quick') is not None for r in results)
    if has_fallback:
        print("\n* = Using quick metrics (CRUMB eval didn't run or failed)")
        print("    Re-run evaluation to get official CRUMB metrics")
    
    # Show summary stats
    valid_ndcg = []
    for r in results:
        ndcg = r.get('nDCG@10') or r.get('ndcg@10_quick')
        if ndcg is not None:
            valid_ndcg.append(ndcg)
    
    if valid_ndcg:
        print(f"\nnDCG@10 statistics:")
        print(f"  Mean: {sum(valid_ndcg)/len(valid_ndcg):.4f}")
        print(f"  Min:  {min(valid_ndcg):.4f}")
        print(f"  Max:  {max(valid_ndcg):.4f}")


def print_by_split_table(results: List[Dict]):
    """Print results grouped by split."""
    if not results:
        print("No results found.")
        return
    
    # Group by split
    by_split = {}
    for r in results:
        split = r['split']
        if split not in by_split:
            by_split[split] = []
        by_split[split].append(r)
    
    print("\n" + "="*140)
    print("RESULTS BY SPLIT")
    print("="*140)
    
    for split in sorted(by_split.keys()):
        split_results = by_split[split]
        print(f"\n{'='*60}")
        print(f"Split: {split} ({len(split_results)} runs)")
        print(f"{'='*60}")
        print(f"\n{'Agent':<20} {'nDCG@10':<10} {'R@100':<10} {'R@1000':<10} {'P@10':<10} {'Chunks/Doc':<12}")
        print("-"*90)
        
        # Sort by nDCG descending
        split_results = sorted(split_results, key=lambda x: (x.get('nDCG@10') or 0), reverse=True)
        
        # Track values for averaging
        sum_ndcg = 0
        sum_r100 = 0
        sum_r1000 = 0
        sum_p10 = 0
        sum_chunks_per_doc = 0
        count_ndcg = 0
        count_r100 = 0
        count_r1000 = 0
        count_p10 = 0
        
        for r in split_results:
            ndcg = r.get('nDCG@10') or r.get('ndcg@10_quick')
            r100 = r.get('R@100') or r.get('recall@100')
            r1000 = r.get('R@1000') or r.get('recall@1000')
            p10 = r.get('P@10')
            
            ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "N/A"
            r100_str = f"{r100:.4f}" if r100 is not None else "N/A"
            r1000_str = f"{r1000:.4f}" if r1000 is not None else "N/A"
            p10_str = f"{p10:.4f}" if p10 is not None else "N/A"
            
            print(
                f"{r['agent']:<20} {ndcg_str:<10} {r100_str:<10} {r1000_str:<10} {p10_str:<10} "
                f"{r['chunks_per_doc']:<12.2f}"
            )
            
            # Accumulate for averages
            sum_chunks_per_doc += r['chunks_per_doc']
            if ndcg is not None:
                sum_ndcg += ndcg
                count_ndcg += 1
            if r100 is not None:
                sum_r100 += r100
                count_r100 += 1
            if r1000 is not None:
                sum_r1000 += r1000
                count_r1000 += 1
            if p10 is not None:
                sum_p10 += p10
                count_p10 += 1
        
        # Print averages row
        print("-"*90)
        n = len(split_results)
        avg_ndcg = sum_ndcg / count_ndcg if count_ndcg > 0 else None
        avg_r100 = sum_r100 / count_r100 if count_r100 > 0 else None
        avg_r1000 = sum_r1000 / count_r1000 if count_r1000 > 0 else None
        avg_p10 = sum_p10 / count_p10 if count_p10 > 0 else None
        avg_chunks_per_doc = sum_chunks_per_doc / n if n > 0 else 0
        
        avg_ndcg_str = f"{avg_ndcg:.4f}" if avg_ndcg is not None else "N/A"
        avg_r100_str = f"{avg_r100:.4f}" if avg_r100 is not None else "N/A"
        avg_r1000_str = f"{avg_r1000:.4f}" if avg_r1000 is not None else "N/A"
        avg_p10_str = f"{avg_p10:.4f}" if avg_p10 is not None else "N/A"
        
        print(
            f"{'AVERAGE':<20} {avg_ndcg_str:<10} {avg_r100_str:<10} {avg_r1000_str:<10} {avg_p10_str:<10} "
            f"{avg_chunks_per_doc:<12.2f}"
        )


def export_to_csv(results: List[Dict], output_path: str):
    """Export results to CSV."""
    import csv
    
    if not results:
        print("No results to export.")
        return
    
    # Get all unique keys
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    fieldnames = sorted(all_keys)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Exported {len(results)} results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and compare evaluation results")
    parser.add_argument("--split", type=str, help="Filter by split name")
    parser.add_argument("--agent", type=str, help="Filter by agent name")
    parser.add_argument("--sort-by", type=str, default="nDCG@10", help="Metric to sort by")
    parser.add_argument("--by-split", action="store_true", help="Group results by split")
    parser.add_argument("--export", type=str, help="Export to CSV file")
    
    args = parser.parse_args()
    
    print(f"\nScanning results from: {RESULTS_DIR}")
    if args.split:
        print(f"  Filter: split = {args.split}")
    if args.agent:
        print(f"  Filter: agent = {args.agent}")
    
    results = load_all_results(split_filter=args.split, agent_filter=args.agent)
    
    if not results:
        print("\n⚠️  No results found.")
        print("\nAvailable splits:")
        if RESULTS_DIR.exists():
            for split_dir in RESULTS_DIR.iterdir():
                if split_dir.is_dir():
                    n_files = len(list(split_dir.glob("*_k100.json")))
                    if n_files > 0:
                        print(f"  - {split_dir.name}: {n_files} runs")
        sys.exit(0)
    
    if args.by_split:
        print_by_split_table(results)
    else:
        print_comparison_table(results, sort_by=args.sort_by)
    
    if args.export:
        export_to_csv(results, args.export)


if __name__ == "__main__":
    main()