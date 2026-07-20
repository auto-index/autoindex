"""
plot_experiment.py — Generate recall@100 and nDCG@10 plots from an experiment directory.

Usage:
    python plot_experiment.py --experiment-dir PATH

The experiment directory must contain:
  - run_journal.json
  - iteration_*_accepted.json  (optional per iteration)
"""

import json
import pathlib
import argparse
from typing import Optional


def _load_journal(experiment_dir: pathlib.Path) -> dict:
    journal_path = experiment_dir / "run_journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"run_journal.json not found in {experiment_dir}")
    with open(journal_path, "r") as f:
        return json.load(f)


def _load_accepted(experiment_dir: pathlib.Path, iteration: int) -> Optional[dict]:
    path = experiment_dir / f"iteration_{iteration}_accepted.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _truncate(text: str, max_len: int = 40) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


_DELTA_KEY_MAP = {
    "recall_at_100": "delta_recall_100",
    "ndcg_at_10": "delta_ndcg_10",
    "recall_at_10": "delta_recall_10",
}


def _make_plot(
    experiment_dir: pathlib.Path,
    iterations: list[dict],
    accepted_data: dict[int, Optional[dict]],
    hypotheses: list[dict],
    metric_key: str,
    metric_label: str,
    filename: str,
) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    # Dark theme colors
    BG_COLOR = "#1e1e1e"
    PANEL_COLOR = "#2a2a2a"
    TEXT_COLOR = "#e0e0e0"
    GRID_COLOR = "#3a3a3a"
    LINE_COLOR = "#7eb8f7"
    ADOPTED_COLOR = "#4caf50"
    NO_CHANGE_COLOR = "#888888"
    ANNOTATION_COLOR = "#f0c040"

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # Build (iteration, h_id) -> delta map so we can shift each point to its
    # post-adoption score.  The iteration record stores the score *before*
    # hypotheses are tested; when a hypothesis is adopted we add its delta so
    # the point reflects what was actually achieved.
    delta_key = _DELTA_KEY_MAP.get(metric_key)
    hyp_delta: dict[tuple[int, str], float] = {}
    if delta_key:
        for h in hypotheses:
            h_it = h.get("iteration")
            h_id = h.get("h_id")
            if h_it is not None and h_id is not None:
                hyp_delta[(h_it, h_id)] = h.get(delta_key, 0.0)

    x_vals = []
    y_vals = []
    adopted_iters = []
    no_change_iters = []

    for entry in iterations:
        it = entry.get("iteration")
        val = entry.get(metric_key)
        if it is None or val is None:
            continue

        adopted_id = entry.get("adopted_hypothesis_id")
        # Adjust score to post-adoption value when a hypothesis was adopted.
        if adopted_id and delta_key:
            val = val + hyp_delta.get((it, adopted_id), 0.0)

        x_vals.append(it)
        y_vals.append(val)

        acc = accepted_data.get(it)
        if adopted_id and acc and acc.get("adopted_hypothesis"):
            adopted_iters.append(it)
        else:
            no_change_iters.append(it)

    if not x_vals:
        raise ValueError(f"No valid data found for metric '{metric_key}'")

    y_by_iter = dict(zip(x_vals, y_vals))

    # Draw connecting line
    ax.plot(
        x_vals,
        y_vals,
        color=LINE_COLOR,
        linewidth=2,
        zorder=2,
        alpha=0.85,
    )

    # Draw no-change points (hollow circles)
    for it in no_change_iters:
        ax.plot(
            it,
            y_by_iter[it],
            marker="o",
            markersize=10,
            markerfacecolor=PANEL_COLOR,
            markeredgecolor=NO_CHANGE_COLOR,
            markeredgewidth=2,
            zorder=3,
        )

    # Draw adopted points (filled circles) and annotations
    annotation_offsets = {}
    n_adopted = len(adopted_iters)
    for idx, it in enumerate(adopted_iters):
        ax.plot(
            it,
            y_by_iter[it],
            marker="o",
            markersize=12,
            markerfacecolor=ADOPTED_COLOR,
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            zorder=4,
        )

        acc = accepted_data.get(it)
        label = ""
        if acc and acc.get("adopted_hypothesis"):
            desc = acc["adopted_hypothesis"].get("description", "")
            label = _truncate(desc, 40)
            if acc.get("synthesized"):
                label += " [synthesized]"

        if label:
            # Alternate annotation direction to reduce overlap
            y_offset = 18 if idx % 2 == 0 else -28
            x_offset = 10

            annotation = ax.annotate(
                label,
                xy=(it, y_by_iter[it]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=8,
                color=ANNOTATION_COLOR,
                ha="left",
                va="bottom" if y_offset > 0 else "top",
                arrowprops=dict(
                    arrowstyle="-",
                    color=ANNOTATION_COLOR,
                    alpha=0.6,
                    lw=1,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#333300",
                    edgecolor=ANNOTATION_COLOR,
                    alpha=0.75,
                    linewidth=0.8,
                ),
                zorder=5,
            )

    # Axis labels and ticks
    ax.set_xlabel("Iteration", color=TEXT_COLOR, fontsize=12, labelpad=8)
    ax.set_ylabel(metric_label, color=TEXT_COLOR, fontsize=12, labelpad=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.tick_params(axis="x", colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)

    # Integer x ticks
    if x_vals:
        ax.set_xticks(x_vals)

    # Grid
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    # Y-axis range with padding
    if y_vals:
        y_min = min(y_vals)
        y_max = max(y_vals)
        y_pad = max((y_max - y_min) * 0.2, 0.05)
        ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    # Title and subtitle
    dir_name = experiment_dir.name
    ax.set_title(
        f"{metric_label} Over Iterations\n{dir_name}",
        color=TEXT_COLOR,
        fontsize=13,
        pad=14,
        loc="left",
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=ADOPTED_COLOR,
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            markersize=10,
            label="Hypothesis adopted",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=PANEL_COLOR,
            markeredgecolor=NO_CHANGE_COLOR,
            markeredgewidth=2,
            markersize=10,
            label="No adoption",
        ),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="lower right",
        facecolor=BG_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout(pad=1.5)

    out_path = experiment_dir / filename
    fig.savefig(out_path, dpi=300, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_experiment(experiment_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    Generate recall@100 and nDCG@10 plots from an experiment directory.

    Parameters
    ----------
    experiment_dir : pathlib.Path
        Directory containing run_journal.json and optional iteration_*_accepted.json files.

    Returns
    -------
    list[pathlib.Path]
        Paths to the saved plot files.
    """
    saved = []

    try:
        journal = _load_journal(experiment_dir)
    except Exception as e:
        print(f"[plot_experiment] Could not load run_journal.json: {e}")
        return saved

    iterations = journal.get("iterations", [])
    if not iterations:
        print("[plot_experiment] No iterations found in run_journal.json")
        return saved

    hypotheses = journal.get("hypotheses", [])

    # Load accepted data for each iteration
    iter_nums = [entry["iteration"] for entry in iterations if "iteration" in entry]
    accepted_data: dict[int, Optional[dict]] = {}
    for it in iter_nums:
        accepted_data[it] = _load_accepted(experiment_dir, it)

    plots = [
        ("recall_at_100",      "Val Recall@100",        "val_recall_at_100.png"),
        ("ndcg_at_10",         "Val nDCG@10",           "val_ndcg_at_10.png"),
        ("eval_recall_at_100", "Eval Recall@100",       "eval_recall_at_100.png"),
        ("eval_ndcg_at_10",    "Eval nDCG@10",          "eval_ndcg_at_10.png"),
    ]
    for metric_key, metric_label, filename in plots:
        try:
            p = _make_plot(
                experiment_dir=experiment_dir,
                iterations=iterations,
                accepted_data=accepted_data,
                hypotheses=hypotheses,
                metric_key=metric_key,
                metric_label=metric_label,
                filename=filename,
            )
            saved.append(p)
            print(f"[plot_experiment] Saved: {p}")
        except Exception as e:
            print(f"[plot_experiment] Failed to generate {filename}: {e}")

    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate recall@100 and nDCG@10 plots from an experiment directory."
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        type=str,
        help="Path to the experiment directory containing run_journal.json",
    )
    args = parser.parse_args()

    try:
        plot_experiment(pathlib.Path(args.experiment_dir))
    except Exception as e:
        print(f"Plot generation failed: {e}")
