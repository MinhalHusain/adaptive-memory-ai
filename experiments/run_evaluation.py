"""Experiment: Run the full evaluation dataset through the NoMemory baseline.

This script validates the evaluation pipeline end-to-end using a MockLLMClient.
It does NOT produce research metrics—only confirms the pipeline works.
"""
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.runner import EvaluationRunner, load_dataset
from src.evaluation.evaluators import DeterministicEvaluator
from src.memory.no_memory import NoMemory
from src.llm.client import MockLLMClient


def main():
    print("=" * 60)
    print("  No-Memory Baseline — Evaluation Pipeline Validation")
    print("=" * 60)

    # --- setup ---
    llm = MockLLMClient(mock_response="I don't have that information.")
    memory = NoMemory()
    evaluator = DeterministicEvaluator()
    runner = EvaluationRunner(llm_client=llm, memory=memory, evaluator=evaluator)

    # --- load dataset ---
    dataset = load_dataset()
    print(f"\nDataset loaded: {len(dataset)} test cases")

    cat_counts = Counter(tc.category for tc in dataset)
    diff_counts = Counter(tc.difficulty for tc in dataset)

    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    print("\nDifficulty distribution:")
    for diff, count in sorted(diff_counts.items()):
        print(f"  {diff}: {count}")

    # --- run ---
    print("\nRunning evaluation...")
    results = runner.run_all(dataset)

    # --- summary ---
    print("\n" + "=" * 60)
    print("  Results Summary (Mock LLM — pipeline validation only)")
    print("=" * 60)
    total = len(results)
    exact = sum(1 for r in results if r.exact_match)
    avg_kw = sum(r.keyword_match_ratio or 0 for r in results) / total if total else 0
    avg_corr = sum(r.correctness or 0 for r in results) / total if total else 0

    print(f"\n  Total cases:               {total}")
    print(f"  Exact matches:             {exact}/{total}")
    print(f"  Avg keyword match ratio:   {avg_kw:.3f}")
    print(f"  Avg correctness:           {avg_corr:.3f}")

    # Breakdown by category
    print("\n  Per-category results:")
    cat_results: dict = {}
    for r in results:
        cat_results.setdefault(r.category, []).append(r)
    for cat in sorted(cat_results):
        cat_list = cat_results[cat]
        cat_exact = sum(1 for r in cat_list if r.exact_match)
        cat_avg = sum(r.correctness or 0 for r in cat_list) / len(cat_list)
        print(f"    {cat}: {cat_exact}/{len(cat_list)} exact, avg_correctness={cat_avg:.3f}")

    # --- export ---
    output_path = Path(__file__).resolve().parent.parent / "results" / "no_memory_baseline_mock.json"
    EvaluationRunner.export_results(results, output_path)
    print(f"\n  Results exported to: {output_path}")

    print("\n" + "=" * 60)
    print("  NOTE: These are mock results validating pipeline correctness.")
    print("  They are NOT research metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
