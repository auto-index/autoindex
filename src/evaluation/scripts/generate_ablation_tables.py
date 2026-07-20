#!/usr/bin/env python3
"""
Generate ablation experiment results tables.

Usage:
    python src/evaluation/scripts/generate_ablation_tables.py
    python src/evaluation/scripts/generate_ablation_tables.py --model "openai/gpt4o"
    python src/evaluation/scripts/generate_ablation_tables.py --condition "agent"
    python src/evaluation/scripts/generate_ablation_tables.py --model "openai/gpt4o" --condition "agent"
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd

def load_all_results(base_dir: Path = Path("ablation_experiments")) -> List[Dict[str, Any]]:
    """Load all results.json files from ablation experiments."""
    results = []
    for results_file in base_dir.rglob("results.json"):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                data['_source_file'] = str(results_file)
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {results_file}: {e}")
    return results

def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    """Compute mean, ignoring None values."""
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)

def aggregate_by_split_and_condition(results: List[Dict[str, Any]], 
                                     model_filter: str = None,
                                     condition_filter: str = None,
                                     exclude_conditions: List[str] = None) -> pd.DataFrame:
    """Aggregate results by split and condition, computing means if multiple runs exist."""
    grouped = defaultdict(list)
    exclude_conditions = exclude_conditions or []
    
    for result in results:
        # Apply filters
        if model_filter and result.get('model') != model_filter:
            continue
        if condition_filter and result.get('condition') != condition_filter:
            continue
        if result.get('condition') in exclude_conditions:
            continue
            
        split = result.get('split', 'unknown')
        condition = result.get('condition', 'unknown')
        model = result.get('model', 'unknown')
        key = (split, condition, model)
        grouped[key].append(result)
    
    rows = []
    for (split, condition, model), runs in grouped.items():
        # Filter out runs with missing critical metrics
        valid_runs = [
            r for r in runs 
            if r.get('baseline_recall_100') is not None 
            and r.get('final_recall_100') is not None
            and r.get('baseline_ndcg_10') is not None
            and r.get('final_ndcg_10') is not None
        ]
        
        if not valid_runs:
            continue
        
        n_runs = len(valid_runs)
        
        row = {
            'model': model,
            'condition': condition,
            'split': split,
            'n_runs': n_runs,
            'baseline_recall_100': safe_mean([r.get('baseline_recall_100') for r in valid_runs]),
            'final_recall_100': safe_mean([r.get('final_recall_100') for r in valid_runs]),
            'baseline_ndcg_10': safe_mean([r.get('baseline_ndcg_10') for r in valid_runs]),
            'final_ndcg_10': safe_mean([r.get('final_ndcg_10') for r in valid_runs]),
        }
        
        # Compute deltas
        row['delta_recall_100'] = row['final_recall_100'] - row['baseline_recall_100']
        row['delta_ndcg_10'] = row['final_ndcg_10'] - row['baseline_ndcg_10']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def format_delta(value: float, format_type: str = 'markdown') -> str:
    """Format delta values with appropriate signs and colors."""
    if format_type == 'markdown':
        if value > 0.001:
            return f"**+{value:.3f}**"
        elif value < -0.001:
            return f"**{value:.3f}**"
        else:
            return f"{value:.3f}"
    elif format_type == 'latex':
        if value > 0.001:
            return f"\\textcolor{{ForestGreen}}{{\\textbf{{+{value:.3f}}}}}"
        elif value < -0.001:
            return f"\\textcolor{{red}}{{\\textbf{{{value:.3f}}}}}"
        else:
            return f"{value:.3f}"
    elif format_type == 'html':
        if value > 0.001:
            color = "green"
            sign = "+"
        elif value < -0.001:
            color = "red"
            sign = ""
        else:
            color = "black"
            sign = "" if value < 0 else "+"
        return f'<span style="color: {color}; font-weight: bold;">{sign}{value:.3f}</span>'
    else:
        return f"{value:+.3f}"

def generate_markdown_table(df: pd.DataFrame, title: str = "Results", show_model: bool = True, show_condition: bool = True) -> str:
    """Generate a Markdown table."""
    # Sort by model, condition, then split
    df = df.sort_values(['model', 'condition', 'split'])
    
    table = f"## {title}\n\n"
    
    # Build header dynamically
    headers = []
    if show_model:
        headers.append("Model")
    if show_condition:
        headers.append("Condition")
    headers.extend(["Split", "Baseline R@100", "Final R@100", "Δ Recall", "Baseline nDCG@10", "Final nDCG@10", "Δ nDCG@10"])
    
    table += "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["-------"] * len(headers)) + "|\n"
    
    for _, row in df.iterrows():
        cells = []
        if show_model:
            cells.append(f"**{row['model']}**")
        if show_condition:
            cells.append(row['condition'])
        cells.extend([
            row['split'],
            f"{row['baseline_recall_100']:.3f}",
            f"{row['final_recall_100']:.3f}",
            format_delta(row['delta_recall_100'], 'markdown'),
            f"{row['baseline_ndcg_10']:.3f}",
            f"{row['final_ndcg_10']:.3f}",
            format_delta(row['delta_ndcg_10'], 'markdown')
        ])
        table += "| " + " | ".join(cells) + " |\n"
    
    table += "\n"
    return table

def generate_latex_table(df: pd.DataFrame, title: str = "Results", label: str = "tab:results", 
                         show_model: bool = True, show_condition: bool = True) -> str:
    """Generate a LaTeX table."""
    # Sort by model, condition, then split
    df = df.sort_values(['model', 'condition', 'split'])
    
    # Determine column count
    n_cols = 6  # metrics columns
    if show_model:
        n_cols += 1
    if show_condition:
        n_cols += 1
    
    col_spec = "|l" * (n_cols - 6) + "|" + "c" * 6 + "|"
    
    table = "% Required LaTeX packages:\n"
    table += "% \\usepackage{xcolor}\n"
    table += "% \\definecolor{ForestGreen}{RGB}{34,139,34}\n\n"
    
    table += "\\begin{table}[ht]\n"
    table += "\\centering\n"
    table += f"\\begin{{tabular}}{{{col_spec}}}\n"
    table += "\\hline\n"
    
    # Header row 1
    headers = []
    if show_model:
        headers.append("\\textbf{Model}")
    if show_condition:
        headers.append("\\textbf{Condition}")
    headers.extend([
        "\\textbf{Split}",
        "\\textbf{Baseline}",
        "\\textbf{Final}",
        "\\textbf{$\\Delta$ Recall}",
        "\\textbf{Baseline}",
        "\\textbf{Final}",
        "\\textbf{$\\Delta$ nDCG@10}"
    ])
    table += " & ".join(headers) + " \\\\\n"
    
    # Header row 2 (sub-headers for metrics)
    subheaders = [""] * (len(headers) - 6)
    subheaders.extend([
        "\\textbf{R@100}",
        "\\textbf{R@100}",
        "",
        "\\textbf{nDCG@10}",
        "\\textbf{nDCG@10}",
        ""
    ])
    table += " & ".join(subheaders) + " \\\\\n"
    table += "\\hline\n"
    
    for _, row in df.iterrows():
        cells = []
        if show_model:
            cells.append(f"\\textbf{{{row['model'].replace('/', '/')}}}")
        if show_condition:
            cells.append(row['condition'].replace('_', '\\_'))
        cells.extend([
            row['split'].replace('_', '\\_'),
            f"{row['baseline_recall_100']:.3f}",
            f"{row['final_recall_100']:.3f}",
            format_delta(row['delta_recall_100'], 'latex'),
            f"{row['baseline_ndcg_10']:.3f}",
            f"{row['final_ndcg_10']:.3f}",
            format_delta(row['delta_ndcg_10'], 'latex')
        ])
        table += " & ".join(cells) + " \\\\\n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += f"\\caption{{{title}}}\n"
    table += f"\\label{{{label}}}\n"
    table += "\\end{table}\n"
    
    return table

def generate_html_table(df: pd.DataFrame, title: str = "Results", show_model: bool = True, show_condition: bool = True) -> str:
    """Generate an HTML table with colored deltas."""
    # Sort by model, condition, then split
    df = df.sort_values(['model', 'condition', 'split'])
    
    table = f"<h2>{title}</h2>\n"
    table += '<table border="1" style="border-collapse: collapse; margin: 20px 0;">\n'
    table += '  <thead>\n'
    table += '    <tr style="background-color: #f2f2f2;">\n'
    
    if show_model:
        table += '      <th style="padding: 8px;">Model</th>\n'
    if show_condition:
        table += '      <th style="padding: 8px;">Condition</th>\n'
    
    table += '      <th style="padding: 8px;">Split</th>\n'
    table += '      <th style="padding: 8px;">Baseline R@100</th>\n'
    table += '      <th style="padding: 8px;">Final R@100</th>\n'
    table += '      <th style="padding: 8px;">Δ Recall</th>\n'
    table += '      <th style="padding: 8px;">Baseline nDCG@10</th>\n'
    table += '      <th style="padding: 8px;">Final nDCG@10</th>\n'
    table += '      <th style="padding: 8px;">Δ nDCG@10</th>\n'
    table += '    </tr>\n'
    table += '  </thead>\n'
    table += '  <tbody>\n'
    
    for _, row in df.iterrows():
        table += '    <tr>\n'
        if show_model:
            table += f'      <td style="padding: 8px;"><strong>{row["model"]}</strong></td>\n'
        if show_condition:
            table += f'      <td style="padding: 8px;">{row["condition"]}</td>\n'
        table += f'      <td style="padding: 8px;"><strong>{row["split"]}</strong></td>\n'
        table += f'      <td style="padding: 8px;">{row["baseline_recall_100"]:.3f}</td>\n'
        table += f'      <td style="padding: 8px;">{row["final_recall_100"]:.3f}</td>\n'
        table += f'      <td style="padding: 8px;">{format_delta(row["delta_recall_100"], "html")}</td>\n'
        table += f'      <td style="padding: 8px;">{row["baseline_ndcg_10"]:.3f}</td>\n'
        table += f'      <td style="padding: 8px;">{row["final_ndcg_10"]:.3f}</td>\n'
        table += f'      <td style="padding: 8px;">{format_delta(row["delta_ndcg_10"], "html")}</td>\n'
        table += '    </tr>\n'
    
    table += '  </tbody>\n'
    table += '</table>\n'
    
    return table

def sanitize_filename(name: str) -> str:
    """Convert model/condition names to valid filenames."""
    return name.replace('/', '_').replace(' ', '_')

def main():
    parser = argparse.ArgumentParser(description="Generate ablation experiment tables")
    parser.add_argument('--format', type=str, default='all', 
                       choices=['markdown', 'latex', 'html', 'csv', 'all'],
                       help='Output format (default: all)')
    parser.add_argument('--model', type=str, default=None,
                       help='Filter by specific model (e.g., "openai/gpt4o")')
    parser.add_argument('--condition', type=str, default=None,
                       help='Filter by specific condition (e.g., "agent")')
    parser.add_argument('--output-dir', type=Path, default=Path('results_tables'),
                       help='Output directory for generated tables')
    parser.add_argument('--exclude-conditions', type=str, nargs='+', 
                       default=['one_shot'],
                       help='Conditions to exclude (default: one_shot)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Load all results
    print("Loading results from ablation_experiments/...")
    results = load_all_results()
    print(f"✓ Loaded {len(results)} results files")
    
    if not results:
        print("Error: No results found!")
        return
    
    # Aggregate data with filters
    df = aggregate_by_split_and_condition(
        results,
        model_filter=args.model,
        condition_filter=args.condition,
        exclude_conditions=args.exclude_conditions
    )
    
    if df.empty:
        print("Error: No data found with specified filters!")
        return
    
    # Determine what to show in columns
    show_model = args.model is None and len(df['model'].unique()) > 1
    show_condition = args.condition is None and len(df['condition'].unique()) > 1
    
    # Build title and filename
    title_parts = []
    suffix_parts = []
    
    if args.model:
        title_parts.append(args.model)
        suffix_parts.append(sanitize_filename(args.model))
    if args.condition:
        title_parts.append(args.condition)
        suffix_parts.append(args.condition)
    
    title = " - ".join(title_parts) if title_parts else "All Results"
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    label = "tab:" + "_".join(suffix_parts) if suffix_parts else "tab:results"
    
    print(f"\n✓ Generating table: {title}")
    print(f"  {len(df)} rows")
    print(f"  {len(df['model'].unique())} model(s): {sorted(df['model'].unique())}")
    print(f"  {len(df['condition'].unique())} condition(s): {sorted(df['condition'].unique())}")
    print(f"  {len(df['split'].unique())} split(s): {sorted(df['split'].unique())}")
    if args.exclude_conditions:
        print(f"  Excluded: {args.exclude_conditions}")
    
    # Generate tables
    if args.format in ['markdown', 'all']:
        output_file = args.output_dir / f"results{suffix}.md"
        with open(output_file, 'w') as f:
            f.write(generate_markdown_table(df, title, show_model, show_condition))
        print(f"\n✓ Saved Markdown: {output_file}")
    
    if args.format in ['latex', 'all']:
        output_file = args.output_dir / f"results{suffix}.tex"
        with open(output_file, 'w') as f:
            f.write(generate_latex_table(df, title, label, show_model, show_condition))
        print(f"✓ Saved LaTeX: {output_file}")
    
    if args.format in ['html', 'all']:
        output_file = args.output_dir / f"results{suffix}.html"
        with open(output_file, 'w') as f:
            f.write(generate_html_table(df, title, show_model, show_condition))
        print(f"✓ Saved HTML: {output_file}")
    
    if args.format in ['csv', 'all']:
        output_file = args.output_dir / f"results{suffix}.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Saved CSV: {output_file}")
    
    print(f"\n✅ Done! Tables saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()