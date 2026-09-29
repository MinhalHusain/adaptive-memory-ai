import json
from pathlib import Path
from collections import Counter

import pytest

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "dataset.json"

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


@pytest.fixture(scope="module")
def dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_dataset_loads(dataset):
    """Dataset is valid JSON and loads as a list."""
    assert isinstance(dataset, list)
    assert len(dataset) > 0


def test_unique_case_ids(dataset):
    """Every case_id is unique."""
    ids = [case["case_id"] for case in dataset]
    assert len(ids) == len(set(ids)), f"Duplicate case_ids: {[i for i, c in Counter(ids).items() if c > 1]}"


def test_valid_categories(dataset):
    """Every case has a valid category."""
    for case in dataset:
        assert "category" in case, f"Missing 'category' in {case.get('case_id', '?')}"
        assert case["category"] in VALID_CATEGORIES, (
            f"Invalid category '{case['category']}' in {case['case_id']}"
        )


def test_all_categories_represented(dataset):
    """All eight categories have at least one case."""
    present = {case["category"] for case in dataset}
    missing = VALID_CATEGORIES - present
    assert not missing, f"Missing categories: {missing}"


def test_valid_difficulty(dataset):
    """Every case has a valid difficulty label."""
    for case in dataset:
        assert "difficulty" in case, f"Missing 'difficulty' in {case['case_id']}"
        assert case["difficulty"] in VALID_DIFFICULTIES, (
            f"Invalid difficulty '{case['difficulty']}' in {case['case_id']}"
        )


def test_has_conversation(dataset):
    """Every case contains a non-empty conversation."""
    for case in dataset:
        assert "conversation" in case, f"Missing 'conversation' in {case['case_id']}"
        assert isinstance(case["conversation"], list), f"'conversation' is not a list in {case['case_id']}"
        assert len(case["conversation"]) >= 2, f"Conversation too short in {case['case_id']}"


def test_has_query(dataset):
    """Every case contains a non-empty query."""
    for case in dataset:
        assert "query" in case, f"Missing 'query' in {case['case_id']}"
        assert isinstance(case["query"], str), f"'query' is not a string in {case['case_id']}"
        assert len(case["query"].strip()) > 0, f"Empty query in {case['case_id']}"


def test_has_expected_information(dataset):
    """Every case contains non-empty expected_information."""
    for case in dataset:
        assert "expected_information" in case, f"Missing 'expected_information' in {case['case_id']}"
        assert isinstance(case["expected_information"], list), (
            f"'expected_information' is not a list in {case['case_id']}"
        )
        assert len(case["expected_information"]) > 0, f"Empty expected_information in {case['case_id']}"


def test_conversation_message_structure(dataset):
    """Every message in every conversation has 'role' and 'content' keys."""
    for case in dataset:
        for i, msg in enumerate(case["conversation"]):
            assert "role" in msg, f"Missing 'role' in message {i} of {case['case_id']}"
            assert "content" in msg, f"Missing 'content' in message {i} of {case['case_id']}"
            assert msg["role"] in ("user", "assistant"), (
                f"Invalid role '{msg['role']}' in message {i} of {case['case_id']}"
            )


def test_no_malformed_records(dataset):
    """Every record contains all required fields with correct types."""
    required_fields = {
        "case_id": str,
        "category": str,
        "conversation": list,
        "query": str,
        "expected_information": list,
        "relevant_memory_ids": list,
        "difficulty": str,
        "notes": str,
    }
    for case in dataset:
        for field_name, field_type in required_fields.items():
            assert field_name in case, f"Missing '{field_name}' in {case.get('case_id', '?')}"
            assert isinstance(case[field_name], field_type), (
                f"'{field_name}' has wrong type in {case['case_id']}: "
                f"expected {field_type.__name__}, got {type(case[field_name]).__name__}"
            )


def test_minimum_scenario_count(dataset):
    """Dataset has at least 48 scenarios."""
    assert len(dataset) >= 48, f"Expected at least 48 scenarios, got {len(dataset)}"


def test_difficulty_distribution(dataset):
    """All three difficulty levels are present."""
    difficulties = {case["difficulty"] for case in dataset}
    assert difficulties == VALID_DIFFICULTIES, f"Missing difficulties: {VALID_DIFFICULTIES - difficulties}"


def test_category_distribution(dataset):
    """Each category has at least 5 scenarios."""
    counts = Counter(case["category"] for case in dataset)
    for cat in VALID_CATEGORIES:
        assert counts[cat] >= 5, f"Category '{cat}' has only {counts[cat]} cases (minimum 5)"
