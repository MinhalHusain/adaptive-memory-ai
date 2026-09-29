import pytest
import datetime
from src.memory.graph_memory import GraphMemory

@pytest.fixture
def memory():
    return GraphMemory()

class TestAddAndRetrieve:
    def test_add_stores_triples(self, memory):
        assert memory.count() == 0
        memory.add("s1", {"user_input": "My name is Alex.", "metadata": {"memory_source_id": "msg_1"}})
        assert memory.count() > 0

    def test_retrieve_returns_relevant_memory(self, memory):
        memory.add("s1", {"user_input": "My name is Alex.", "metadata": {"memory_source_id": "msg_1"}})
        memory.add("s1", {"user_input": "I live in Berlin.", "metadata": {"memory_source_id": "msg_2"}})
        
        results = memory.retrieve("What is my name?")
        assert len(results) > 0
        assert "Alex" in results[0]["content"]
        assert results[0]["memory_source_id"] == "msg_1"
        assert "metadata" in results[0]

    def test_retrieve_empty_store(self, memory):
        results = memory.retrieve("Hello?")
        assert len(results) == 0

class TestUpdateAndForget:
    def test_update_changes_content(self, memory):
        memory.add("s1", {"user_input": "My name is Alex.", "metadata": {"memory_source_id": "msg_1"}})
        results = memory.retrieve("name")
        assert len(results) > 0
        mem_id = results[0]["memory_id"]
        
        memory.update(mem_id, {"content": "My name is Alexander.", "metadata": {"memory_source_id": "msg_1"}})
        results2 = memory.retrieve("name")
        assert len(results2) > 0
        assert "Alexander" in results2[0]["content"]

    def test_forget_removes_memory(self, memory):
        memory.add("s1", {"user_input": "I like pizza.", "metadata": {"memory_source_id": "msg_1"}})
        results = memory.retrieve("pizza")
        assert len(results) > 0
        mem_id = results[0]["memory_id"]
        
        memory.forget(mem_id)
        results2 = memory.retrieve("pizza")
        assert len(results2) == 0

class TestReset:
    def test_reset_clears_graph(self, memory):
        memory.add("s1", {"user_input": "I like pizza."})
        assert memory.count() > 0
        memory.reset()
        assert memory.count() == 0
