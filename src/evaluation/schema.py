from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class EvaluationRecord:
    """Schema for tracking evaluation metrics of a single interaction."""
    question_id: str
    category: str
    difficulty: str
    expected_information: List[str]
    relevant_memory_ids: List[str]
    retrieved_memories: List[Dict[str, Any]]
    generated_answer: str

    # Deterministic evaluation results
    exact_match: Optional[bool] = None
    keyword_matches: Optional[Dict[str, bool]] = None
    keyword_match_ratio: Optional[float] = None

    # Semantic evaluation results (populated by evaluators)
    semantic_similarity: Optional[float] = None
    llm_judge_score: Optional[float] = None
    llm_judge_reasoning: Optional[str] = None

    # Timing and token metrics
    correctness: Optional[float] = None
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Retrieval metrics (ID-based, Ground Truth)
    recall_at_1: Optional[float] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    precision_at_1: Optional[float] = None
    precision_at_3: Optional[float] = None
    precision_at_5: Optional[float] = None

    # Retrieval metrics (Diagnostic/Content-Proxy)
    legacy_keyword_recall: Optional[float] = None
    legacy_keyword_precision: Optional[float] = None

    # Conversation quality metrics (future)
    long_term_consistency: Optional[float] = None
    personalization_score: Optional[float] = None

    # Memory quality metrics (future)
    redundancy_score: Optional[float] = None
    conflict_count: Optional[int] = None
    outdated_memory_count: Optional[int] = None
    memory_growth: Optional[int] = None

    # Efficiency metrics (future)
    context_tokens: Optional[int] = None
    memory_size: Optional[int] = None
    retrieval_latency_ms: Optional[float] = None
    cost: Optional[float] = None

    # Metadata
    memory_type: Optional[str] = None
    embedding_model: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class TestCase:
    """Schema for a single evaluation test case loaded from the dataset."""
    case_id: str
    category: str
    conversation: List[Dict[str, str]]
    query: str
    expected_information: List[str]
    relevant_memory_ids: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        return cls(
            case_id=data["case_id"],
            category=data["category"],
            conversation=data["conversation"],
            query=data["query"],
            expected_information=data["expected_information"],
            relevant_memory_ids=data.get("relevant_memory_ids", []),
            difficulty=data.get("difficulty", "medium"),
            notes=data.get("notes", ""),
        )


VALID_CATEGORIES = {
    "FACT_RETENTION",
    "LONG_RANGE_RETRIEVAL",
    "INFORMATION_UPDATE",
    "CONFLICTING_INFORMATION",
    "IRRELEVANT_INFORMATION",
    "REDUNDANT_INFORMATION",
    "MULTI_HOP_RELATIONSHIP",
    "PERSONALIZATION",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
