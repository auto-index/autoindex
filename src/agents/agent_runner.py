"""
agent_runner.py – Abstract base class for LLM-driven preprocessing agents.

Each agent subclass implements build_prompt() and call_llm(), then calls
run(n_loops) to start the iterative eval-improve loop.
"""

from __future__ import annotations

import hashlib
import json
import sys
import importlib.util
import pathlib
from abc import ABC, abstractmethod

_PROJECT_ROOT = pathlib.Path(__file__).parents[2]


class AgentRunner(ABC):
    agent_name: str
    split: str = "tip_of_the_tongue"
    baseline_results: dict | None = None
    _system_instruction: str = ""

    def run(self, n_loops: int) -> None:
        """Main eval-improve loop."""
        preprocess_path = (
            _PROJECT_ROOT / "src" / "agents" / self.agent_name / "preprocess.py"
        )

        # Compute baseline by running baseline preprocessor on current corpus
        print(f"\n{'#'*60}")
        print(f"# Baseline (raw documents, no preprocessing) — computed dynamically")
        print(f"{'#'*60}")
        self.baseline_results = self._compute_baseline()
        print(f"  Recall@100 : {self.baseline_results['recall_at_k']:.4f}")
        print(f"  nDCG@10    : {self.baseline_results['ndcg']:.4f}")
        self.on_baseline_complete(self.baseline_results)

        for i in range(n_loops):
            print(f"\n{'#'*60}")
            print(f"# Iteration {i + 1} / {n_loops}")
            print(f"{'#'*60}")

            prompt = None
            eval_results = None
            
            if not preprocess_path.read_text(encoding="utf-8").strip():
                print("[agent_runner] preprocess.py is empty, skipping eval.")
                prompt = "[agent_runner] No eval results available. Please write a preprocess() function that chunks documents."
            else:
                try:
                    raw_results = self.run_eval(iteration=i)
                    # Flatten results for prompt builder (expects old format)
                    eval_results = {
                        "top_k": raw_results["config"]["top_k"],
                        "recall_at_k": raw_results["metrics"]["recall_at_100"],
                        "ndcg": raw_results["metrics"]["ndcg_at_10"],
                        "n_queries": raw_results["config"]["n_queries"],
                        "n_chunks": raw_results["config"]["n_chunks"],
                        "n_docs": raw_results["config"]["n_docs"],
                        "chunks_per_doc": raw_results["config"]["chunks_per_doc"],
                    }
                    prompt = self.build_prompt(iteration=i, eval_results=eval_results)
                except Exception as e:
                    print(f"[agent_runner] Eval failed (iteration {i + 1}): {e}")
                    import traceback
                    traceback.print_exc()
                    prompt = f"[agent_runner] Eval failed with error: {e}\nPlease fix the preprocess() function."
            
            # Safety check - ensure prompt is never None
            if prompt is None:
                prompt = f"[agent_runner] Iteration {i+1}: No prompt generated. Please write preprocessing code."
            
            self.call_llm(prompt=prompt, iteration=i)

        # Run final eval to show results of the last generated preprocess.py
        print(f"\n{'#'*60}")
        print(f"# Final eval (after {n_loops} loop{'s' if n_loops != 1 else ''})")
        print(f"{'#'*60}")
        if not preprocess_path.read_text(encoding="utf-8").strip():
            print("[agent_runner] preprocess.py is empty, skipping final eval.")
        else:
            try:
                self.run_eval()
            except Exception as e:
                print(f"[agent_runner] Final eval failed: {e}")

    def run_eval(self, iteration: int = None) -> dict:  # ADD iteration parameter
        """
        Dynamically load Preprocessor from the agent's preprocess.py and run
        the static evaluate() harness. Returns the results dict.
        
        Args:
            iteration: Current iteration number (0-indexed), used for file naming
        """
        eval_scripts_dir = _PROJECT_ROOT / "src" / "evaluation" / "scripts"
        eval_dir = _PROJECT_ROOT / "src" / "evaluation"
        src_dir = _PROJECT_ROOT / "src"

        for p in [str(eval_scripts_dir), str(eval_dir), str(src_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Import evaluate() from the static harness
        from test_preprocessing_split import evaluate
        from schema import Document as EvalDocument
        from base import BasePreprocessor

        # Reload preprocess.py fresh each iteration so code changes take effect
        preprocess_path = (
            _PROJECT_ROOT / "src" / "agents" / self.agent_name / "preprocess.py"
        )
        spec = importlib.util.spec_from_file_location(
            f"_agent_{self.agent_name}_preprocess", preprocess_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        inner_preprocessor = module.Preprocessor()

        # Wrap the agent's preprocessor to replace doc_ids with opaque hashes before
        # the agent code sees them.  This prevents exploitation of the path-encoded
        # query terms present in CRUMB stack_exchange doc_ids (e.g.
        # "yeast_dissolve_in_sugar/Osmosis.txt" → a fixed hex token).
        # chunk.doc_id values are remapped back to originals before scoring.
        class _DocIdSanitizer(BasePreprocessor):
            name = inner_preprocessor.name
            description = inner_preprocessor.description

            def preprocess(self, docs):
                id_map = {
                    d.doc_id: "doc_" + hashlib.sha256(d.doc_id.encode()).hexdigest()[:16]
                    for d in docs
                }
                reverse_map = {v: k for k, v in id_map.items()}
                sanitized = [
                    EvalDocument(doc_id=id_map[d.doc_id], text=d.text, metadata=d.metadata)
                    for d in docs
                ]
                chunks = inner_preprocessor.preprocess(sanitized)
                for chunk in chunks:
                    if chunk.doc_id in reverse_map:
                        chunk.doc_id = reverse_map[chunk.doc_id]
                return chunks

        preprocessor = _DocIdSanitizer()

        return evaluate(
            preprocessor,
            split=self.split,
            top_k=100,
            save_results=True,
            iteration=iteration,
            track_iterations=True
        )

    def _compute_baseline(self) -> dict:
        """Run baseline preprocessor on current data and return results dict.

        Subclasses (e.g. AnalysisCodeAgent) override this to use the BM25
        server instead of the static harness.
        """
        eval_scripts_dir = _PROJECT_ROOT / "src" / "evaluation" / "scripts"
        eval_dir = _PROJECT_ROOT / "src" / "evaluation"
        agents_dir = _PROJECT_ROOT / "src" / "agents"

        for p in [str(eval_scripts_dir), str(eval_dir), str(agents_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from test_preprocessing_split import evaluate
        from baseline.preprocess import Preprocessor as BaselinePreprocessor

        baseline = BaselinePreprocessor()
        results = evaluate(baseline, split=self.split, top_k=100)
        return {
            "recall_at_k": results["metrics"]["recall_at_100"],
            "ndcg": results["metrics"]["ndcg_at_10"],
            "query_results": results.get("query_results", []),
        }

    def on_baseline_complete(self, baseline_results: dict) -> None:
        """Called after baseline eval; override to inject baseline numbers into system instruction."""

    @abstractmethod
    def build_prompt(self, iteration: int, eval_results: dict | None) -> str: ...

    @abstractmethod
    def call_llm(self, prompt: str, iteration: int) -> None: ...
