import json
import time
from pathlib import Path
from typing import List, Optional

from src.conversation.pipeline import ConversationalPipeline
from src.conversation.session import ConversationSession
from src.evaluation.evaluators import BaseEvaluator, DeterministicEvaluator
from src.evaluation.schema import EvaluationRecord, TestCase
from src.llm.client import BaseLLMClient
from src.memory.base import BaseMemory
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "dataset.json"


def load_dataset(path: Optional[Path] = None) -> List[TestCase]:
    """Load evaluation test cases from the dataset JSON file."""
    path = path or DATASET_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TestCase.from_dict(item) for item in raw]


class EvaluationRunner:
    """Runs evaluation scenarios through a pluggable memory pipeline.

    Design:
        For each test case the runner:
        1. Resets the memory (clear any state from the previous case).
        2. Replays the scripted conversation in Session A, populating memory.
        3. Issues the evaluation query in a NEW Session B, so the LLM has
           no within-session context from the conversation — only whatever
           the memory system retrieves.
        4. Evaluates the generated answer against expected information.

    This design ensures we are testing *long-term memory*, not the
    LLM's context-window recall.

    Usage:
        runner = EvaluationRunner(
            llm_client=my_llm,
            memory=NoMemory(),           # or VectorMemory(), etc.
            evaluator=DeterministicEvaluator(),
        )
        results = runner.run_all()
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        memory: BaseMemory,
        evaluator: Optional[BaseEvaluator] = None,
        dataset_path: Optional[Path] = None,
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        self.llm_client = llm_client
        self.memory = memory
        self.evaluator = evaluator or DeterministicEvaluator()
        self.dataset_path = dataset_path
        self.system_prompt = system_prompt

    def run_single(self, case: TestCase) -> EvaluationRecord:
        """Execute a single test case and return an EvaluationRecord."""
        logger.info(f"Running case {case.case_id} [{case.category}]")

        # --- reset memory so cases are independent ---
        self.memory.reset()

        pipeline = ConversationalPipeline(
            llm_client=self.llm_client,
            memory=self.memory,
            system_prompt=self.system_prompt,
        )

        # --- Phase 1: replay conversation in Session A (populate memory) ---
        conversation_session = ConversationSession(
            session_id=f"conv-{case.case_id}"
        )
        for msg in case.conversation:
            if msg["role"] == "user":
                metadata = {}
                if "memory_source_id" in msg:
                    metadata["memory_source_id"] = msg["memory_source_id"]
                pipeline.process_message(msg["content"], conversation_session, metadata=metadata)
            elif msg["role"] == "assistant":
                # Overwrite the LLM-generated reply with the scripted one
                # so conversation history matches the dataset exactly.
                history = conversation_session.get_history()
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = msg["content"]

        # --- Phase 2: query in Session B (fresh, no conversation history) ---
        query_session = ConversationSession(
            session_id=f"query-{case.case_id}"
        )

        start = time.perf_counter()
        generated_answer = pipeline.process_message(case.query, query_session)
        latency = (time.perf_counter() - start) * 1000  # ms

        # Capture what the pipeline actually retrieved for this query
        retrieved_memories = pipeline.last_retrieved_memories

        # Get embedding model — walk into sub-stores for HybridMemory
        emb = getattr(self.memory, "embedding", None)
        if emb is None:
            # HybridMemory: look inside vector_memory
            vm = getattr(self.memory, "vector_memory", None)
            if vm:
                emb = getattr(vm, "embedding", None)
        emb_name = None
        if emb:
            emb_name = getattr(emb, "model_name", type(emb).__name__)

        # --- build record ---
        record = EvaluationRecord(
            question_id=case.case_id,
            category=case.category,
            difficulty=case.difficulty,
            expected_information=case.expected_information,
            relevant_memory_ids=case.relevant_memory_ids,
            retrieved_memories=retrieved_memories,
            generated_answer=generated_answer,
            latency_ms=round(latency, 2),
            memory_type=type(self.memory).__name__,
            embedding_model=emb_name,
            memory_size=getattr(self.memory, "count", lambda: None)(),
            notes=case.notes,
        )

        # --- evaluate ---
        record = self.evaluator.evaluate(record)

        logger.info(
            f"  -> case {case.case_id}: correctness={record.correctness}, "
            f"exact_match={record.exact_match}, "
            f"keyword_ratio={record.keyword_match_ratio}, "
            f"memories_retrieved={len(retrieved_memories)}"
        )
        return record

    def run_all(self, dataset: Optional[List[TestCase]] = None) -> List[EvaluationRecord]:
        """Run every test case in the dataset and return all records."""
        if dataset is None:
            dataset = load_dataset(self.dataset_path)

        logger.info(f"Starting evaluation run: {len(dataset)} cases, memory={type(self.memory).__name__}")
        results: List[EvaluationRecord] = []
        for case in dataset:
            record = self.run_single(case)
            results.append(record)

        # --- summary ---
        total = len(results)
        avg_correctness = sum(r.correctness or 0 for r in results) / total if total else 0
        exact_matches = sum(1 for r in results if r.exact_match)
        logger.info(
            f"Evaluation complete: {total} cases, "
            f"avg_correctness={avg_correctness:.3f}, "
            f"exact_matches={exact_matches}/{total}"
        )
        return results

    @staticmethod
    def export_results(results: List[EvaluationRecord], path: Path) -> None:
        """Export evaluation results to a JSON file for manual inspection."""
        import dataclasses
        serializable = [dataclasses.asdict(r) for r in results]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Results exported to {path}")
