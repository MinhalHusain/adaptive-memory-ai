from src.evaluation.runner import EvaluationRunner, load_dataset
from src.evaluation.evaluators import DeterministicEvaluator
from src.memory.no_memory import NoMemory
from src.llm.client import MockLLMClient


def test_runner_loads_dataset():
    dataset = load_dataset()
    assert len(dataset) > 0
    assert dataset[0].case_id is not None


def test_runner_single_case():
    llm = MockLLMClient(mock_response="Mock answer")
    memory = NoMemory()
    runner = EvaluationRunner(llm_client=llm, memory=memory, evaluator=DeterministicEvaluator())
    dataset = load_dataset()
    record = runner.run_single(dataset[0])
    assert record.question_id == dataset[0].case_id
    assert record.generated_answer == "Mock answer"
    assert record.memory_type == "NoMemory"


def test_runner_full_run():
    llm = MockLLMClient(mock_response="Mock answer")
    memory = NoMemory()
    runner = EvaluationRunner(llm_client=llm, memory=memory, evaluator=DeterministicEvaluator())
    results = runner.run_all()
    assert len(results) > 0
    for r in results:
        assert r.generated_answer == "Mock answer"
        assert r.memory_type == "NoMemory"
        assert r.correctness is not None
