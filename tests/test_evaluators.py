import pytest
from src.evaluation.evaluators import DeterministicEvaluator, SemanticEvaluator, CompositeEvaluator
from src.evaluation.schema import EvaluationRecord


def _make_record(generated_answer: str, expected_information: list) -> EvaluationRecord:
    return EvaluationRecord(
        question_id="test",
        category="FACT_RETENTION",
        difficulty="easy",
        expected_information=expected_information,
        relevant_memory_ids=[],
        retrieved_memories=[],
        generated_answer=generated_answer,
    )


class TestDeterministicEvaluator:
    def test_exact_match_all_present(self):
        record = _make_record("I study computer science at MIT", ["computer science", "MIT"])
        evaluator = DeterministicEvaluator()
        result = evaluator.evaluate(record)
        assert result.exact_match is True
        assert result.correctness == 1.0
        assert result.keyword_match_ratio == 1.0

    def test_exact_match_none_present(self):
        record = _make_record("I don't know", ["computer science", "MIT"])
        evaluator = DeterministicEvaluator()
        result = evaluator.evaluate(record)
        assert result.exact_match is False
        assert result.correctness == 0.0

    def test_partial_match(self):
        record = _make_record("I study at MIT", ["computer science", "MIT"])
        evaluator = DeterministicEvaluator()
        result = evaluator.evaluate(record)
        assert result.exact_match is False
        # MIT is matched as keyword, "computer science" keywords partially match
        assert result.keyword_match_ratio > 0.0

    def test_case_insensitive(self):
        record = _make_record("I study COMPUTER SCIENCE at mit", ["computer science", "MIT"])
        evaluator = DeterministicEvaluator()
        result = evaluator.evaluate(record)
        assert result.exact_match is True


class TestSemanticEvaluator:
    def test_fallback_to_deterministic(self):
        record = _make_record("I study computer science at MIT", ["computer science", "MIT"])
        evaluator = SemanticEvaluator(llm_client=None)
        result = evaluator.evaluate(record)
        assert result.exact_match is True
        assert result.correctness == 1.0


class TestCompositeEvaluator:
    def test_chains_evaluators(self):
        record = _make_record("I study computer science at MIT", ["computer science", "MIT"])
        evaluator = CompositeEvaluator([DeterministicEvaluator(), SemanticEvaluator()])
        result = evaluator.evaluate(record)
        assert result.exact_match is True
        assert result.correctness == 1.0


class TestRetrievalEvaluator:
    def test_id_based_metrics(self):
        from src.evaluation.evaluators import RetrievalEvaluator
        
        record = _make_record("answer", ["info"])
        record.relevant_memory_ids = ["mem_1", "mem_2"]
        
        # Simulating retrieving mem_3, mem_1, mem_4, mem_2
        record.retrieved_memories = [
            {"metadata": {"memory_source_id": "mem_3"}},
            {"metadata": {"memory_source_id": "mem_1"}},
            {"metadata": {"memory_source_id": "mem_4"}},
            {"metadata": {"memory_source_id": "mem_2"}},
        ]
        
        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate(record)
        
        # relevant = [mem_1, mem_2], len = 2
        # retrieved = [mem_3, mem_1, mem_4, mem_2]
        
        # @1: retrieved=[mem_3], hits=0
        assert result.recall_at_1 == 0.0
        assert result.precision_at_1 == 0.0
        
        # @3: retrieved=[mem_3, mem_1, mem_4], hits=1 (mem_1)
        assert result.recall_at_3 == 0.5   # 1/2
        assert result.precision_at_3 == round(1/3, 4)
        
        # @5: retrieved=[mem_3, mem_1, mem_4, mem_2], hits=2
        assert result.recall_at_5 == 1.0   # 2/2
        assert result.precision_at_5 == 0.5   # 2/4

    def test_missing_source_id_fallback(self):
        from src.evaluation.evaluators import RetrievalEvaluator
        record = _make_record("answer", ["info"])
        record.relevant_memory_ids = ["runtime_id_1"]
        record.retrieved_memories = [
            {"memory_id": "runtime_id_1"} # No metadata
        ]
        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate(record)
        assert result.recall_at_1 == 1.0
