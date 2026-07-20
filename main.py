"""
main.py – CLI entry point for AutoIndex agents.

Usage:
    # Full AutoIndex loop (Analysis Agent + Code Agent + search history):
    uv run python main.py --agent analysis_code_agent --loops 5 --condition agent_contrastive --split tip_of_the_tongue

    # Ablation conditions:
    uv run python main.py --agent analysis_code_agent --loops 5 --condition agent_contrastive_no_history
    uv run python main.py --agent analysis_code_agent --loops 5 --condition agent_noinput

    # One-shot LLM baseline (no loops):
    uv run python main.py --agent one_shot --split tip_of_the_tongue
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an AutoIndex representation-program agent."
    )
    parser.add_argument(
        "--agent",
        default="analysis_code_agent",
        choices=[
            "analysis_code_agent",
            "one_shot",
        ],
        help="Which agent to run (default: analysis_code_agent)",
    )
    parser.add_argument(
        "--condition",
        default="agent_contrastive",
        choices=["agent", "agent_history", "agent_contrastive", "agent_contrastive_no_history", "agent_noinput"],
        help="Ablation condition for analysis_code_agent (default: agent_contrastive)",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Number of eval+improve loops (default: 3)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="tip_of_the_tongue",
        help="CRUMB split name (default: tip_of_the_tongue)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model for analysis_code_agent and one_shot (e.g. gemini/gemini-2.5-pro)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        dest="api_base",
        help="Override API base URL (omit to use provider default, e.g. for native Gemini API)",
    )
    parser.add_argument(
        "--max-distractors",
        type=int,
        default=9000,
        help="Max non-relevant docs to sample"
    )

    args = parser.parse_args()

    # Reset analysis_code_agent/preprocess.py to clean baseline at the start of every run.
    import shutil, pathlib
    repo_root = pathlib.Path(__file__).parent
    clean = repo_root / "src" / "agents" / "baseline" / "preprocess.py"
    target = repo_root / "src" / "agents" / "analysis_code_agent" / "preprocess.py"
    shutil.copy(clean, target)
    print(f"[main] Reset preprocess.py from clean baseline: {clean}")

    if args.agent == "analysis_code_agent":
        from src.agents.analysis_code_agent import AnalysisCodeAgent
        use_history = args.condition in ("agent_history", "agent_contrastive")
        use_contrastive = args.condition in ("agent_contrastive", "agent_contrastive_no_history")
        use_analysis = args.condition != "agent_noinput"
        agent = AnalysisCodeAgent(
            use_history=use_history,
            use_contrastive=use_contrastive,
            use_analysis=use_analysis,
            model=args.model,
            api_base=args.api_base,
        )

    elif args.agent == "one_shot":
        from src.agents.analysis_code_agent.one_shot_agent import run_one_shot
        run_one_shot(split=args.split, model=args.model, api_base=args.api_base, max_distractors=args.max_distractors)
        return

    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    agent.split = args.split
    agent.run(n_loops=args.loops)


if __name__ == "__main__":
    main()
