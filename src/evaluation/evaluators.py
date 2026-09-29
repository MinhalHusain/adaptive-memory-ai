from abc import ABC, abstractmethod
from typing import List

from src.evaluation.schema import EvaluationRecord


class BaseEvaluator(ABC):
    """Abstract interface for answer evaluators.

    All evaluators take an EvaluationRecord (which already contains the
    generated_answer and expected_information) and return a *new* record
    with evaluation fields populated.  This keeps evaluation logic
    decoupled from generation logic.
    """

    @abstractmethod
    def evaluate(self, record: EvaluationRecord) -> EvaluationRecord:
        """Score a single evaluation record and return an updated copy."""
        pass


class DeterministicEvaluator(BaseEvaluator):
    """Layered deterministic evaluation using exact match and keyword
    presence checks.  No LLM calls are required.
    """

    def evaluate(self, record: EvaluationRecord) -> EvaluationRecord:
        answer_lower = record.generated_answer.lower()
        expected = record.expected_information

        # --- exact match (all expected items appear verbatim) ---
        all_exact = all(info.lower() in answer_lower for info in expected)
        record.exact_match = all_exact

        # --- keyword / entity matching ---
        keyword_hits: dict[str, bool] = {}
        for info in expected:
            # Check each keyword token from the expected info
            tokens = info.lower().split()
            keyword_hits[info] = any(tok in answer_lower for tok in tokens)

        record.keyword_matches = keyword_hits
        matched = sum(1 for v in keyword_hits.values() if v)
        record.keyword_match_ratio = matched / len(keyword_hits) if keyword_hits else 0.0

        # --- composite correctness (deterministic) ---
        if all_exact:
            record.correctness = 1.0
        else:
            record.correctness = record.keyword_match_ratio

        return record


class SemanticEvaluator(BaseEvaluator):
    """Extensible interface for semantic / LLM-based evaluation.

    This is a placeholder implementation that future phases will extend
    to use an LLM judge (e.g. GPT-4 or Gemini) for nuanced scoring.
    The current implementation falls back to deterministic evaluation
    so that the pipeline works end-to-end without API keys.
    """

    def __init__(self, llm_client=None):
        self._llm_client = llm_client
        self._fallback = DeterministicEvaluator()

    def evaluate(self, record: EvaluationRecord) -> EvaluationRecord:
        if self._llm_client is None:
            # Fall back to deterministic scoring when no LLM is available
            return self._fallback.evaluate(record)

        # --- Future: LLM-as-judge scoring ---
        # Construct a prompt asking the LLM to score the answer against
        # expected_information and populate:
        #   record.semantic_similarity
        #   record.llm_judge_score
        #   record.llm_judge_reasoning
        # For now, defer to deterministic evaluation.
        return self._fallback.evaluate(record)


class CompositeEvaluator(BaseEvaluator):
    """Runs multiple evaluators in sequence, accumulating results."""

    def __init__(self, evaluators: List[BaseEvaluator]):
        self.evaluators = evaluators

    def evaluate(self, record: EvaluationRecord) -> EvaluationRecord:
        for evaluator in self.evaluators:
            record = evaluator.evaluate(record)
        return record


class RetrievalEvaluator(BaseEvaluator):
    """Evaluates memory retrieval quality.

    Computes:
    - Memory count / size statistics.
    - Recall@K and Precision@K **when ground-truth relevant_memory_ids
      are available** in the dataset.  Since the current dataset does not
      yet have ground-truth memory IDs, these fields are left as ``None``
      and the raw retrieval outputs are preserved for later annotation.

    Important: this evaluator measures *retrieval* quality, not *answer*
    quality.  Use DeterministicEvaluator / SemanticEvaluator for answer
    scoring.
    """

    def evaluate(self, record: EvaluationRecord) -> EvaluationRecord:
        retrieved = record.retrieved_memories or []
        relevant_ids = set(record.relevant_memory_ids or [])
        
        # Calculate Recall@K and Precision@K
        # The retrieved memories are already ordered by similarity (Top 1 is at index 0)
        # We need to look at the 'memory_source_id' from the metadata
        retrieved_source_ids = []
        for mem in retrieved:
            meta = mem.get("metadata", {})
            # Get the source id if it exists, otherwise use runtime memory_id to avoid crash
            # but it won't match anyway which is correct for missing metadata
            src_id = meta.get("memory_source_id", mem.get("memory_id"))
            retrieved_source_ids.append(src_id)
            
        def calc_recall(k: int) -> float:
            if not relevant_ids:
                return 0.0
            top_k_ids = retrieved_source_ids[:k]
            hits = sum(1 for rid in relevant_ids if rid in top_k_ids)
            return round(hits / len(relevant_ids), 4)
            
        def calc_precision(k: int) -> float:
            top_k_ids = retrieved_source_ids[:k]
            if not top_k_ids:
                return 0.0
            hits = sum(1 for ret_id in top_k_ids if ret_id in relevant_ids)
            return round(hits / len(top_k_ids), 4)

        if relevant_ids:
            record.recall_at_1 = calc_recall(1)
            record.recall_at_3 = calc_recall(3)
            record.recall_at_5 = calc_recall(5)
            record.precision_at_1 = calc_precision(1)
            record.precision_at_3 = calc_precision(3)
            record.precision_at_5 = calc_precision(5)
        
        # -- Legacy content-based retrieval proxy check --
        if retrieved and record.expected_information:
            hits = 0
            for exp_info in record.expected_information:
                exp_lower = exp_info.lower()
                for mem in retrieved:
                    if exp_lower in mem.get("content", "").lower():
                        hits += 1
                        break
            total_expected = len(record.expected_information)
            record.legacy_keyword_recall = round(hits / total_expected, 4) if total_expected else 0.0
            record.legacy_keyword_precision = round(hits / len(retrieved), 4) if retrieved else 0.0

        return record

