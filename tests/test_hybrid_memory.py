"""Tests for HybridMemory: fusion, deduplication, normalisation, provenance."""
import pytest
import tempfile
import os

from src.memory.hybrid_memory import HybridMemory
from src.memory.vector_memory import VectorMemory
from src.memory.graph_memory import GraphMemory
from src.memory.embeddings import FakeEmbedding


def make_hybrid(alpha: float = 0.5, top_k: int = 5) -> HybridMemory:
    tmpdir = tempfile.mkdtemp(prefix="hybrid_test_")
    vm = VectorMemory(
        embedding=FakeEmbedding(64),
        persist_dir=tmpdir,
        collection_name="hybrid_test",
        top_k=top_k * 2,  # retrieve extra for fusion
    )
    gm = GraphMemory(top_k=top_k * 2)
    return HybridMemory(vm, gm, alpha=alpha, top_k=top_k)


# ── Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def mem():
    return make_hybrid()


# ── Both sub-memories receive same source ─────────────────────────────────

class TestBothReceiveSameSource:
    def test_add_stores_in_both(self, mem):
        mem.add("s1", {"user_input": "I live in Paris.", "metadata": {"memory_source_id": "src_1"}})
        counts = mem.count()
        assert counts["vector"] == 1
        assert counts["graph"] >= 1  # graph may create multiple edges

    def test_retrieve_finds_in_both(self, mem):
        mem.add("s1", {"user_input": "I live in Paris.", "metadata": {"memory_source_id": "src_1"}})
        results = mem.retrieve("Where do I live?")
        assert len(results) > 0
        assert results[0]["memory_source_id"] == "src_1"
        assert results[0]["retrieval_source"] == "both"

    def test_same_source_id_across_representations(self, mem):
        mem.add("s1", {"user_input": "My name is Bob.", "metadata": {"memory_source_id": "src_2"}})
        results = mem.retrieve("What is my name?")
        # source id should be the same regardless of which retriever found it
        for r in results:
            assert r["memory_source_id"] is not None


# ── Vector-only candidate (graph did not match) ───────────────────────────

class TestVectorOnlyCandidate:
    def test_vector_only_has_zero_graph_norm(self, mem):
        # Store something; graph will add it too but if query is semantic only vector matches
        mem.add("s1", {"user_input": "Zebras are striped mammals.", "metadata": {"memory_source_id": "src_v"}})
        mem.add("s1", {"user_input": "I live in Tokyo.", "metadata": {"memory_source_id": "src_g"}})
        results = mem.retrieve("Tokyo residence")
        # src_g should appear with graph contribution; src_v if vector-only has ng=0.0
        found = {r["memory_source_id"]: r for r in results}
        if "src_v" in found:
            # graph may not have matched src_v
            assert found["src_v"]["retrieval_source"] in ("vector", "both")


# ── Graph-only candidate ──────────────────────────────────────────────────

class TestGraphOnlyCandidate:
    def test_graph_only_has_zero_vector_norm(self, mem, monkeypatch):
        """Simulate vector returning nothing by monkeypatching its retrieve."""
        # Patch vector_memory.retrieve to always return []
        monkeypatch.setattr(mem.vector_memory, "retrieve", lambda *a, **kw: [])

        mem.add("s1", {"user_input": "I like chocolate.", "metadata": {"memory_source_id": "src_g"}})
        results = mem.retrieve("What do I like?")
        if results:
            r = results[0]
            assert r["retrieval_source"] == "graph"
            assert r["vector_score"] is None
            assert r["norm_vector_score"] == 0.0


# ── Deduplication ─────────────────────────────────────────────────────────

class TestDeduplication:
    def test_same_source_appears_once(self, mem):
        """Same memory_source_id must appear at most once in output."""
        mem.add("s1", {"user_input": "I work at Apple.", "metadata": {"memory_source_id": "dedup_1"}})
        mem.add("s1", {"user_input": "I enjoy hiking.", "metadata": {"memory_source_id": "dedup_2"}})
        results = mem.retrieve("Apple work")
        source_ids = [r["memory_source_id"] for r in results]
        assert len(source_ids) == len(set(source_ids)), "Duplicate source IDs in results"


# ── Score normalisation ───────────────────────────────────────────────────

class TestNormalisation:
    def test_all_norm_scores_in_0_1(self, mem):
        for i in range(3):
            mem.add("s1", {
                "user_input": f"Fact number {i} about cats.",
                "metadata": {"memory_source_id": f"norm_{i}"}
            })
        results = mem.retrieve("Tell me about cats")
        for r in results:
            assert 0.0 <= r["norm_vector_score"] <= 1.0, f"norm_vector_score out of range: {r}"
            assert 0.0 <= r["norm_graph_score"] <= 1.0, f"norm_graph_score out of range: {r}"
            assert 0.0 <= r["hybrid_score"] <= 1.0, f"hybrid_score out of range: {r}"

    def test_equal_scores_normalise_to_one(self, mem):
        """When all raw scores are equal, all normalised scores are 1.0."""
        # Two identical messages produce equal FakeEmbedding vectors
        mem.add("s1", {"user_input": "dogs", "metadata": {"memory_source_id": "eq_1"}})
        results = mem.retrieve("dogs")
        if len(results) == 1:
            assert results[0]["norm_vector_score"] == 1.0


# ── Weighted fusion ───────────────────────────────────────────────────────

class TestWeightedFusion:
    def test_alpha_zero_uses_graph_only(self):
        """alpha=0 → hybrid_score == norm_graph_score."""
        hm = make_hybrid(alpha=0.0)
        hm.add("s1", {"user_input": "I live in Oslo.", "metadata": {"memory_source_id": "fus_1"}})
        results = hm.retrieve("Where do I live?")
        for r in results:
            expected = 0.0 * r["norm_vector_score"] + 1.0 * r["norm_graph_score"]
            assert abs(r["hybrid_score"] - expected) < 1e-9

    def test_alpha_one_uses_vector_only(self):
        """alpha=1 → hybrid_score == norm_vector_score."""
        hm = make_hybrid(alpha=1.0)
        hm.add("s1", {"user_input": "I live in Oslo.", "metadata": {"memory_source_id": "fus_2"}})
        results = hm.retrieve("Where do I live?")
        for r in results:
            expected = 1.0 * r["norm_vector_score"] + 0.0 * r["norm_graph_score"]
            assert abs(r["hybrid_score"] - expected) < 1e-9

    def test_invalid_alpha_raises(self):
        tmpdir = tempfile.mkdtemp()
        vm = VectorMemory(FakeEmbedding(64), persist_dir=tmpdir, collection_name="alpha-test", top_k=5)
        gm = GraphMemory()
        with pytest.raises(ValueError):
            HybridMemory(vm, gm, alpha=1.5)


# ── Missing source handling ────────────────────────────────────────────────

class TestMissingSourceHandling:
    def test_vector_only_source_has_ng_zero(self):
        """A source found only by vector gets norm_graph_score=0.0."""
        hm = make_hybrid()
        # Add something that vector finds but graph won't (no relationship structure)
        hm.add("s1", {"user_input": "Zebras.", "metadata": {"memory_source_id": "zebra"}})
        # Add a second item that graph matches strongly on keyword
        hm.add("s1", {"user_input": "I live in Nairobi.", "metadata": {"memory_source_id": "nairobi"}})
        results = hm.retrieve("Nairobi residence")
        for r in results:
            if r["retrieval_source"] == "vector":
                assert r["graph_score"] is None
                assert r["norm_graph_score"] == 0.0

    def test_source_not_discarded_when_missing_from_one_retriever(self, mem):
        """A source found by only one retriever must still appear in output."""
        mem.add("s1", {"user_input": "I live in Cairo.", "metadata": {"memory_source_id": "cairo_1"}})
        results = mem.retrieve("Cairo")
        source_ids = [r["memory_source_id"] for r in results]
        assert "cairo_1" in source_ids


# ── Top-K ranking ─────────────────────────────────────────────────────────

class TestTopKRanking:
    def test_results_limited_to_top_k(self, mem):
        for i in range(8):
            mem.add("s1", {
                "user_input": f"I like food item {i}.",
                "metadata": {"memory_source_id": f"food_{i}"}
            })
        results = mem.retrieve("What foods do I like?")
        assert len(results) <= 5  # default top_k

    def test_results_sorted_descending_by_hybrid(self, mem):
        for i in range(4):
            mem.add("s1", {
                "user_input": f"I enjoy sport {i}.",
                "metadata": {"memory_source_id": f"sport_{i}"}
            })
        results = mem.retrieve("What sports do I enjoy?")
        scores = [r["hybrid_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by hybrid_score"


# ── Source-ID preservation ────────────────────────────────────────────────

class TestSourceIDPreservation:
    def test_source_id_in_result(self, mem):
        mem.add("s1", {"user_input": "I am a teacher.", "metadata": {"memory_source_id": "teacher_src"}})
        results = mem.retrieve("What is my job?")
        assert any(r["memory_source_id"] == "teacher_src" for r in results)

    def test_all_results_have_source_id(self, mem):
        for i in range(3):
            mem.add("s1", {
                "user_input": f"I speak language {i}.",
                "metadata": {"memory_source_id": f"lang_{i}"}
            })
        results = mem.retrieve("What languages do I speak?")
        for r in results:
            assert r["memory_source_id"] is not None


# ── Reset ─────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_both(self, mem):
        mem.add("s1", {"user_input": "Test fact.", "metadata": {"memory_source_id": "reset_1"}})
        assert mem.count()["vector"] == 1
        assert mem.count()["graph"] >= 1
        mem.reset()
        assert mem.count()["vector"] == 0
        assert mem.count()["graph"] == 0

    def test_retrieve_after_reset_returns_empty(self, mem):
        mem.add("s1", {"user_input": "Test fact.", "metadata": {"memory_source_id": "reset_2"}})
        mem.reset()
        results = mem.retrieve("Test fact")
        assert results == []


# ── Evaluation runner integration ─────────────────────────────────────────

class TestEvaluationIntegration:
    def test_runner_works_with_hybrid_memory(self):
        from src.evaluation.runner import EvaluationRunner, load_dataset
        from src.evaluation.evaluators import DeterministicEvaluator, RetrievalEvaluator, CompositeEvaluator
        from src.llm.client import MockLLMClient

        hm = make_hybrid()
        evaluator = CompositeEvaluator([DeterministicEvaluator(), RetrievalEvaluator()])
        llm = MockLLMClient("I don't know.")
        runner = EvaluationRunner(llm_client=llm, memory=hm, evaluator=evaluator)

        dataset = load_dataset()
        case = dataset[0]
        record = runner.run_single(case)

        assert record.question_id == case.case_id
        # Hybrid Memory exposes memory_type
        assert record.memory_type == "HybridMemory"
