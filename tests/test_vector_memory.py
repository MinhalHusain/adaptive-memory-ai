"""Tests for VectorMemory.

All tests use FakeEmbedding + a temporary directory so they run
offline without paid API keys.
"""

import tempfile
import pytest

from src.memory.vector_memory import VectorMemory
from src.memory.embeddings import FakeEmbedding
from src.conversation.pipeline import ConversationalPipeline
from src.conversation.session import ConversationSession
from src.evaluation.runner import EvaluationRunner, load_dataset
from src.evaluation.evaluators import DeterministicEvaluator, RetrievalEvaluator, CompositeEvaluator
from src.llm.client import MockLLMClient


@pytest.fixture
def embedding():
    return FakeEmbedding(dim=64)


@pytest.fixture
def memory(tmp_path, embedding):
    return VectorMemory(
        embedding=embedding,
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_memories",
        top_k=5,
    )


# ── 1. Adding a memory ───────────────────────────────────────────────────

class TestAdd:
    def test_add_stores_memory(self, memory):
        assert memory.count() == 0
        memory.add("s1", {"user_input": "My name is Alex"})
        assert memory.count() == 1

    def test_add_multiple_memories(self, memory):
        memory.add("s1", {"user_input": "My name is Alex"})
        memory.add("s1", {"user_input": "I like pizza"})
        memory.add("s2", {"user_input": "I work at Google"})
        assert memory.count() == 3

    def test_add_ignores_empty_input(self, memory):
        memory.add("s1", {"user_input": ""})
        memory.add("s1", {"user_input": "   "})
        memory.add("s1", {})
        assert memory.count() == 0


# ── 2. Retrieving a relevant memory ──────────────────────────────────────

class TestRetrieve:
    def test_retrieve_returns_relevant_memory(self, memory):
        memory.add("s1", {"user_input": "My name is Alex"})
        memory.add("s1", {"user_input": "The weather is sunny today"})
        results = memory.retrieve("What is my name?")
        assert len(results) > 0
        # The name-related memory should be ranked first
        assert "Alex" in results[0]["content"]

    def test_retrieve_empty_store_returns_empty(self, memory):
        results = memory.retrieve("hello")
        assert results == []

    def test_retrieve_result_structure(self, memory):
        memory.add("s1", {"user_input": "I study computer science"})
        results = memory.retrieve("What do I study?")
        assert len(results) == 1
        r = results[0]
        assert "memory_id" in r
        assert "content" in r
        assert "similarity" in r
        assert "metadata" in r
        assert r["content"] == "I study computer science"
    def test_metadata_preservation(self, memory):
        memory.add("s1", {
            "user_input": "I like python",
            "metadata": {"memory_source_id": "test_msg_1", "extra_val": 42}
        })
        results = memory.retrieve("python")
        assert len(results) == 1
        meta = results[0]["metadata"]
        assert meta["memory_source_id"] == "test_msg_1"
        assert meta["extra_val"] == 42
        assert "timestamp" in meta
        
class TestSentenceTransformerInit:
    def test_init_sentence_transformer(self):
        try:
            from src.memory.embeddings import SentenceTransformerEmbedding
            # Initializing it will load the model, which might be slow on first download,
            # but usually fast if cached. We just check if it imports and instantiates without error.
            emb = SentenceTransformerEmbedding()
            assert emb.dimension > 0
            # Also test single embed
            vec = emb.embed_single("test")
            assert len(vec) == emb.dimension
        except ImportError:
            pytest.skip("sentence-transformers not installed")


# ── 3. Similarity ranking ────────────────────────────────────────────────

class TestSimilarityRanking:
    def test_more_similar_text_ranks_higher(self, memory):
        memory.add("s1", {"user_input": "My favorite color is blue"})
        memory.add("s1", {"user_input": "I enjoy cooking Italian pasta"})
        memory.add("s1", {"user_input": "What is my favorite color?"})

        results = memory.retrieve("Tell me about my favorite color")
        # The color-related memories should rank above the cooking one
        contents = [r["content"] for r in results]
        color_idx = next(
            i for i, c in enumerate(contents) if "color" in c.lower()
        )
        cooking_idx = next(
            i for i, c in enumerate(contents) if "cooking" in c.lower()
        )
        assert color_idx < cooking_idx

    def test_similarity_scores_are_ordered(self, memory):
        memory.add("s1", {"user_input": "Python programming is fun"})
        memory.add("s1", {"user_input": "The cat sat on the mat"})
        memory.add("s1", {"user_input": "I love Python and machine learning"})
        results = memory.retrieve("Tell me about Python programming")
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)


# ── 4. Top-K behavior ────────────────────────────────────────────────────

class TestTopK:
    def test_default_top_k(self, memory):
        for i in range(10):
            memory.add("s1", {"user_input": f"Memory number {i}"})
        results = memory.retrieve("memory")
        assert len(results) == 5  # default top_k

    def test_custom_top_k(self, memory):
        for i in range(10):
            memory.add("s1", {"user_input": f"Memory number {i}"})
        results = memory.retrieve("memory", top_k=3)
        assert len(results) == 3

    def test_top_k_larger_than_store(self, memory):
        memory.add("s1", {"user_input": "Only one memory"})
        results = memory.retrieve("memory", top_k=10)
        assert len(results) == 1


# ── 5. Updating a memory ─────────────────────────────────────────────────

class TestUpdate:
    def test_update_changes_content(self, memory):
        memory.add("s1", {"user_input": "My favorite language is Python"})
        results = memory.retrieve("favorite language")
        memory_id = results[0]["memory_id"]

        memory.update(memory_id, {"content": "My favorite language is Rust"})
        results = memory.retrieve("favorite language")
        assert "Rust" in results[0]["content"]


# ── 6. Forgetting / deleting a memory ────────────────────────────────────

class TestForget:
    def test_forget_removes_memory(self, memory):
        memory.add("s1", {"user_input": "Secret info"})
        assert memory.count() == 1
        results = memory.retrieve("secret")
        memory_id = results[0]["memory_id"]
        memory.forget(memory_id)
        assert memory.count() == 0

    def test_forget_one_of_many(self, memory):
        memory.add("s1", {"user_input": "Keep this"})
        memory.add("s1", {"user_input": "Delete this"})
        assert memory.count() == 2
        results = memory.retrieve("delete")
        target_id = results[0]["memory_id"]
        memory.forget(target_id)
        assert memory.count() == 1


# ── 7. Persistence across instances ──────────────────────────────────────

class TestPersistence:
    def test_memories_survive_new_instance(self, embedding, tmp_path):
        persist_dir = str(tmp_path / "persist_test")

        mem1 = VectorMemory(
            embedding=embedding,
            persist_dir=persist_dir,
            collection_name="persist",
        )
        mem1.add("s1", {"user_input": "My name is Alex and I live in Paris"})
        assert mem1.count() == 1

        # Create a completely new instance pointing to the same directory
        mem2 = VectorMemory(
            embedding=embedding,
            persist_dir=persist_dir,
            collection_name="persist",
        )
        assert mem2.count() == 1
        results = mem2.retrieve("What is my name?")
        assert len(results) == 1
        assert "Alex" in results[0]["content"]


# ── 8. No cross-contamination ────────────────────────────────────────────

class TestIsolation:
    def test_reset_clears_all_memories(self, memory):
        memory.add("s1", {"user_input": "Data from old session"})
        assert memory.count() == 1
        memory.reset()
        assert memory.count() == 0
        results = memory.retrieve("old session")
        assert results == []


# ── 9. Integration with pipeline ─────────────────────────────────────────

class TestPipelineIntegration:
    def test_pipeline_stores_and_retrieves(self, memory):
        llm = MockLLMClient(mock_response="Mock response")
        pipeline = ConversationalPipeline(llm_client=llm, memory=memory)

        session1 = ConversationSession("s1")
        pipeline.process_message("My name is Alex", session1)
        pipeline.process_message("I like apples", session1)

        # New session — pipeline should retrieve memories
        session2 = ConversationSession("s2")
        pipeline.process_message("What do I like?", session2)

        # Verify memory was retrieved and injected into LLM context
        last_messages = llm.last_messages
        # Should contain: system prompt + memory context + user message
        assert len(last_messages) >= 2
        # At least one message should mention relevant info
        all_content = " ".join(m["content"] for m in last_messages)
        assert "apples" in all_content.lower() or "Alex" in all_content

    def test_pipeline_no_memory_with_no_memory_class(self):
        from src.memory.no_memory import NoMemory
        llm = MockLLMClient(mock_response="Mock response")
        pipeline = ConversationalPipeline(llm_client=llm, memory=NoMemory())

        session1 = ConversationSession("s1")
        pipeline.process_message("My name is Alex", session1)

        session2 = ConversationSession("s2")
        pipeline.process_message("What is my name?", session2)

        # NoMemory: LLM should only get system prompt + user message
        last_messages = llm.last_messages
        assert len(last_messages) == 2  # system + user


# ── 10. EvaluationRunner compatibility ────────────────────────────────────

class TestEvaluationRunnerCompatibility:
    def test_runner_works_with_vector_memory(self, memory):
        llm = MockLLMClient(mock_response="Alex studies computer science at MIT")
        evaluator = CompositeEvaluator([
            DeterministicEvaluator(),
            RetrievalEvaluator(),
        ])
        runner = EvaluationRunner(
            llm_client=llm,
            memory=memory,
            evaluator=evaluator,
        )
        dataset = load_dataset()
        record = runner.run_single(dataset[0])
        assert record.question_id == dataset[0].case_id
        assert record.memory_type == "VectorMemory"
        assert record.generated_answer is not None

    def test_runner_full_run_with_vector_memory(self, memory):
        llm = MockLLMClient(mock_response="Mock answer")
        runner = EvaluationRunner(
            llm_client=llm,
            memory=memory,
            evaluator=DeterministicEvaluator(),
        )
        # Run on a small subset to keep tests fast
        dataset = load_dataset()[:3]
        results = runner.run_all(dataset)
        assert len(results) == 3
        for r in results:
            assert r.memory_type == "VectorMemory"
