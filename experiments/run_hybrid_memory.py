"""Experiment: Run the full 50-case evaluation through HybridMemory.

Configuration via environment variables
----------------------------------------
EMBEDDING_PROVIDER   sentence-transformer | fake  (default: fake)
HYBRID_ALPHA         float in [0,1]               (default: 0.5)
VECTOR_TOP_K         int                           (default: 5)
VECTOR_STORE_DIR     path (default: tmp dir, cleared per run)
OPENAI_API_KEY       if set, uses OpenAI LLM

Results are exported to results/hybrid_memory_preliminary.json.
All metrics are PRELIMINARY and labelled as such.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.evaluation.runner import EvaluationRunner, load_dataset
from src.evaluation.evaluators import (
    CompositeEvaluator,
    DeterministicEvaluator,
    RetrievalEvaluator,
)
from src.memory.hybrid_memory import HybridMemory
from src.memory.vector_memory import VectorMemory
from src.memory.graph_memory import GraphMemory
from src.memory.embeddings import FakeEmbedding
from src.llm.client import MockLLMClient, OpenAILLMClient


def main() -> None:
    load_dotenv()

    print("=" * 64)
    print("  Hybrid Memory Baseline - Preliminary Evaluation")
    print("=" * 64)

    # ── Fusion weight ───────────────────────────────────────────────
    alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
    if not 0.0 <= alpha <= 1.0:
        print(f"ERROR: HYBRID_ALPHA must be in [0,1], got {alpha}")
        sys.exit(1)
    print(f"\nFusion: alpha={alpha}  (vector weight={alpha}, graph weight={1 - alpha})")
    print("NOTE: alpha is NOT tuned against the evaluation dataset.")

    # ── Embedding ───────────────────────────────────────────────────
    provider = os.getenv("EMBEDDING_PROVIDER", "fake")
    if provider == "sentence-transformer":
        try:
            from src.memory.embeddings import SentenceTransformerEmbedding
            embedding = SentenceTransformerEmbedding()
            emb_label = f"SentenceTransformer ({embedding.model_name})"
        except ImportError:
            print("ERROR: EMBEDDING_PROVIDER=sentence-transformer but sentence-transformers not installed.")
            print("Run: pip install sentence-transformers")
            sys.exit(1)
    elif provider == "fake":
        embedding = FakeEmbedding(dim=64)
        emb_label = f"FakeEmbedding (dim={embedding.dimension})"
    else:
        print(f"ERROR: Unknown EMBEDDING_PROVIDER '{provider}'.")
        sys.exit(1)
    print(f"Embedding: {emb_label}")

    # ── LLM ─────────────────────────────────────────────────────────
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

    # ── Sub-systems ─────────────────────────────────────────────────
    top_k = int(os.getenv("VECTOR_TOP_K", "5"))
    persist_dir = os.getenv("VECTOR_STORE_DIR", None)
    if persist_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="hybrid_eval_")
        persist_dir = tmpdir
        print(f"Vector store: temporary ({persist_dir})")
    else:
        print(f"Vector store: {persist_dir}")

    vector_memory = VectorMemory(
        embedding=embedding,
        persist_dir=persist_dir,
        collection_name="hybrid_eval_memories",
        top_k=top_k * 2,   # retrieve extra for fusion
    )
    graph_memory = GraphMemory(top_k=top_k * 2)
    memory = HybridMemory(
        vector_memory=vector_memory,
        graph_memory=graph_memory,
        alpha=alpha,
        top_k=top_k,
    )
    print(f"Top-K (after fusion): {top_k}")

    # ── Evaluator ───────────────────────────────────────────────────
    evaluator = CompositeEvaluator([
        DeterministicEvaluator(),
        RetrievalEvaluator(),
    ])

    # ── Dataset ─────────────────────────────────────────────────────
    dataset = load_dataset()
    print(f"\nDataset: {len(dataset)} test cases")

    # ── Run ─────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    runner = EvaluationRunner(
        llm_client=llm,
        memory=memory,
        evaluator=evaluator,
    )
    results = runner.run_all(dataset)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  PRELIMINARY Results (Hybrid Memory Baseline)")
    print("=" * 64)

    total = len(results)

    # A. Retrieval metrics
    avg_r1  = sum(r.recall_at_1    or 0 for r in results) / total
    avg_r3  = sum(r.recall_at_3    or 0 for r in results) / total
    avg_r5  = sum(r.recall_at_5    or 0 for r in results) / total
    avg_p1  = sum(r.precision_at_1 or 0 for r in results) / total
    avg_p3  = sum(r.precision_at_3 or 0 for r in results) / total
    avg_p5  = sum(r.precision_at_5 or 0 for r in results) / total
    avg_legacy = sum(r.legacy_keyword_recall or 0 for r in results) / total

    # B. Answer metrics
    exact    = sum(1 for r in results if r.exact_match)
    avg_kw   = sum(r.keyword_match_ratio or 0 for r in results) / total
    avg_corr = sum(r.correctness or 0 for r in results) / total

    # C. Efficiency
    avg_lat      = sum(r.latency_ms or 0 for r in results) / total
    avg_retr     = sum(len(r.retrieved_memories or []) for r in results) / total
    # memory size = total edges+docs across both sub-stores at end of each case
    # memory size: for hybrid, count() returns dict; runner stores it directly.
    # memory_size field may be a dict (hybrid) or int (vector/graph).
    def _to_int(v):
        if isinstance(v, dict):
            return sum(v.values())
        return v or 0
    avg_mem_size = sum(_to_int(r.memory_size) for r in results) / total

    print("\n  A. RETRIEVAL METRICS (Ground-Truth ID-based)")
    print(f"     Recall@1:    {avg_r1:.3f}   | Precision@1:    {avg_p1:.3f}")
    print(f"     Recall@3:    {avg_r3:.3f}   | Precision@3:    {avg_p3:.3f}")
    print(f"     Recall@5:    {avg_r5:.3f}   | Precision@5:    {avg_p5:.3f}")
    print(f"     [Legacy Proxy Recall]: {avg_legacy:.3f}")

    print("\n  B. ANSWER METRICS")
    print(f"     Exact matches:       {exact}/{total}")
    print(f"     Keyword match ratio: {avg_kw:.3f}")
    print(f"     Answer correctness:  {avg_corr:.3f}")
    if isinstance(llm, MockLLMClient):
        print("     * NOTE: Using MockLLMClient — answer-level metrics are not research results.")

    print("\n  C. EFFICIENCY METRICS")
    print(f"     Avg Pipeline Latency: {avg_lat:.2f} ms")
    print(f"     Avg Memories Retr.:   {avg_retr:.2f}")
    print(f"     Avg Memory DB Size:   {avg_mem_size:.1f}")

    # Per-category
    print("\n  Per-category breakdown (Recall@5 / Correctness):")
    cat_results: dict = {}
    for r in results:
        cat_results.setdefault(r.category, []).append(r)
    for cat in sorted(cat_results):
        cat_list = cat_results[cat]
        n = len(cat_list)
        cat_corr   = sum(r.correctness or 0 for r in cat_list) / n
        cat_recall5 = sum(r.recall_at_5 or 0 for r in cat_list) / n
        print(f"    {cat}: R@5={cat_recall5:.3f} | Corr={cat_corr:.3f}")

    # ── Export ──────────────────────────────────────────────────────
    if provider == "sentence-transformer":
        out_filename = "hybrid_memory_real_embedding.json"
    else:
        out_filename = "hybrid_memory_preliminary.json"

    output_path = (
        Path(__file__).resolve().parent.parent / "results" / out_filename
    )
    # Attach full experiment config to each record
    config_note = (
        f"alpha={alpha} | embedding={emb_label} | top_k={top_k} | "
        f"vector_store={persist_dir} | graph=NetworkX-2hop"
    )
    for r in results:
        r.notes = config_note

    EvaluationRunner.export_results(results, output_path)
    print(f"\n  Results exported to: {output_path}")

    print("\n" + "=" * 64)
    print("  NOTE: These are PRELIMINARY results.")
    print("  Do not treat as final research metrics.")
    print("=" * 64)


if __name__ == "__main__":
    main()
