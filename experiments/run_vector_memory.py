"""Experiment: Run the full evaluation dataset through VectorMemory.

This uses FakeEmbedding + MockLLMClient by default for pipeline
validation.  When OPENAI_API_KEY is set, it can optionally use
real embeddings and a real LLM.

Results are PRELIMINARY and labeled as such.
"""
import os
import sys
import tempfile
from pathlib import Path
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.evaluation.runner import EvaluationRunner, load_dataset
from src.evaluation.evaluators import (
    DeterministicEvaluator,
    RetrievalEvaluator,
    CompositeEvaluator,
)
from src.memory.vector_memory import VectorMemory
from src.memory.embeddings import FakeEmbedding
from src.llm.client import MockLLMClient, OpenAILLMClient


def main():
    load_dotenv()

    print("=" * 64)
    print("  Vector Memory Baseline - Preliminary Evaluation")
    print("=" * 64)

    # --- Embedding ---
    # Try sentence-transformers first, fall back to FakeEmbedding ONLY if explicitly requested
    embedding = None
    provider = os.getenv("EMBEDDING_PROVIDER", "fake")
    if provider == "sentence-transformer":
        try:
            from src.memory.embeddings import SentenceTransformerEmbedding
            embedding = SentenceTransformerEmbedding()
            print(f"\nEmbedding: SentenceTransformer ({embedding.model_name})")
        except ImportError:
            print("\nERROR: EMBEDDING_PROVIDER is 'sentence-transformer' but sentence-transformers is not installed.")
            print("Run: pip install sentence-transformers")
            sys.exit(1)
    elif provider == "fake":
        embedding = FakeEmbedding(dim=64)
        print(f"\nEmbedding: FakeEmbedding (dim={embedding.dimension})")
    else:
        print(f"\nERROR: Unknown EMBEDDING_PROVIDER '{provider}'.")
        sys.exit(1)

    # --- LLM ---
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm = OpenAILLMClient()
            print(f"LLM: OpenAI ({llm.model})")
        except Exception as e:
            print(f"OpenAI init failed ({e}), using MockLLMClient.")
            llm = MockLLMClient(mock_response="I don't have that information.")
    else:
        print("LLM: MockLLMClient (no OPENAI_API_KEY)")
        llm = MockLLMClient(mock_response="I don't have that information.")

    # --- Vector Memory ---
    persist_dir = os.getenv("VECTOR_STORE_DIR", None)
    if persist_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="vector_eval_")
        persist_dir = tmpdir
        print(f"Vector store: temporary directory ({persist_dir})")
    else:
        print(f"Vector store: {persist_dir}")

    top_k = int(os.getenv("VECTOR_TOP_K", "5"))
    memory = VectorMemory(
        embedding=embedding,
        persist_dir=persist_dir,
        collection_name="eval_memories",
        top_k=top_k,
    )
    print(f"Top-K: {top_k}")

    # --- Evaluator ---
    evaluator = CompositeEvaluator([
        DeterministicEvaluator(),
        RetrievalEvaluator(),
    ])

    # --- Dataset ---
    dataset = load_dataset()
    print(f"\nDataset: {len(dataset)} test cases")

    cat_counts = Counter(tc.category for tc in dataset)
    diff_counts = Counter(tc.difficulty for tc in dataset)

    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    print("\nDifficulty distribution:")
    for diff, count in sorted(diff_counts.items()):
        print(f"  {diff}: {count}")

    # --- Run ---
    print("\nRunning evaluation...")
    runner = EvaluationRunner(
        llm_client=llm,
        memory=memory,
        evaluator=evaluator,
    )
    results = runner.run_all(dataset)

    # --- Summary ---
    print("\n" + "=" * 64)
    print("  PRELIMINARY Results (Vector Memory Baseline)")
    print("=" * 64)
    
    total = len(results)
    
    # 1. Answer Metrics
    exact = sum(1 for r in results if r.exact_match)
    avg_kw = sum(r.keyword_match_ratio or 0 for r in results) / total if total else 0
    avg_corr = sum(r.correctness or 0 for r in results) / total if total else 0
    
    # 2. Retrieval Metrics (ID-based)
    avg_r1 = sum(r.recall_at_1 or 0 for r in results) / total if total else 0
    avg_r3 = sum(r.recall_at_3 or 0 for r in results) / total if total else 0
    avg_r5 = sum(r.recall_at_5 or 0 for r in results) / total if total else 0
    avg_p1 = sum(r.precision_at_1 or 0 for r in results) / total if total else 0
    avg_p3 = sum(r.precision_at_3 or 0 for r in results) / total if total else 0
    avg_p5 = sum(r.precision_at_5 or 0 for r in results) / total if total else 0
    
    # Legacy proxy metrics for reference
    avg_legacy_r = sum(r.legacy_keyword_recall or 0 for r in results) / total if total else 0
    
    # 3. Efficiency Metrics
    avg_lat = sum(r.latency_ms or 0 for r in results) / total if total else 0
    avg_mems = sum(len(r.retrieved_memories or []) for r in results) / total if total else 0
    avg_mem_size = sum(r.memory_size or 0 for r in results) / total if total else 0

    print("\n  A. RETRIEVAL METRICS (Ground-Truth ID-based)")
    print(f"     Recall@1:    {avg_r1:.3f}   | Precision@1:    {avg_p1:.3f}")
    print(f"     Recall@3:    {avg_r3:.3f}   | Precision@3:    {avg_p3:.3f}")
    print(f"     Recall@5:    {avg_r5:.3f}   | Precision@5:    {avg_p5:.3f}")
    print(f"     [Legacy Proxy Recall]: {avg_legacy_r:.3f}")

    print("\n  B. ANSWER METRICS")
    print(f"     Exact matches:       {exact}/{total}")
    print(f"     Keyword match ratio: {avg_kw:.3f}")
    print(f"     Answer correctness:  {avg_corr:.3f}")
    if isinstance(llm, MockLLMClient):
        print("     * NOTE: Using MockLLMClient. Answer-level research evaluation is pending real LLM configuration.")
        
    print("\n  C. EFFICIENCY METRICS")
    print(f"     Avg Pipeline Latency: {avg_lat:.2f} ms")
    print(f"     Avg Memories Retr.:   {avg_mems:.2f}")
    print(f"     Avg Memory DB Size:   {avg_mem_size:.1f}")

    # Per-category
    print("\n  Per-category breakdown (Recall@5 / Correctness):")
    cat_results: dict = {}
    for r in results:
        cat_results.setdefault(r.category, []).append(r)
    for cat in sorted(cat_results):
        cat_list = cat_results[cat]
        n = len(cat_list)
        cat_corr = sum(r.correctness or 0 for r in cat_list) / n
        cat_recall5 = sum(r.recall_at_5 or 0 for r in cat_list) / n
        print(f"    {cat}: R@5={cat_recall5:.3f} | Corr={cat_corr:.3f}")

    # --- Export ---
    # Choose filename based on embedding to preserve the preliminary run
    if provider == "sentence-transformer":
        filename = "vector_memory_real_embedding.json"
    else:
        filename = "vector_memory_preliminary.json"
        
    output_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / filename
    )
    EvaluationRunner.export_results(results, output_path)
    print(f"\n  Results exported to: {output_path}")

    print("\n" + "=" * 64)
    print("  NOTE: These are PRELIMINARY results.")
    print("  Do not treat as final research metrics.")
    print("=" * 64)


if __name__ == "__main__":
    main()
