import pytest
from src.memory.graph_memory import GraphMemory

@pytest.fixture
def memory():
    return GraphMemory(top_k=5)

class TestMultiHopRetrieval:
    def test_one_hop_retrieval(self, memory):
        memory.add("s1", {"user_input": "My friend is Alex.", "metadata": {"memory_source_id": "msg_1"}})
        results = memory.retrieve("Who is my friend?")
        assert len(results) > 0
        assert "Alex" in results[0]["content"]

    def test_two_hop_retrieval(self, memory):
        # A -> B
        memory.add("s1", {"user_input": "My friend is Alex.", "metadata": {"memory_source_id": "msg_1"}})
        # B -> C
        memory.add("s1", {"user_input": "Alex lives in Berlin.", "metadata": {"memory_source_id": "msg_2"}})
        
        # Query requires knowing Alex is the friend to find Berlin.
        # Actually, query "Where does my friend live?"
        # Keywords: "friend", "live"
        # Entry node: "User's friend" (has "friend")
        # 1-hop: ("User's friend", "is", "Alex") -> hop1 node: "Alex"
        # 2-hop: ("Alex", "lives in", "Berlin") -> returns this edge!
        
        results = memory.retrieve("Where does my friend live?")
        
        retrieved_texts = [r["content"] for r in results]
        
        assert any("Berlin" in text for text in retrieved_texts), "Failed to retrieve 2-hop connection"
        
    def test_conflict_resolution_single_valued(self, memory):
        memory.add("s1", {"user_input": "I live in Seattle.", "metadata": {"memory_source_id": "msg_1"}})
        memory.add("s1", {"user_input": "I live in Portland.", "metadata": {"memory_source_id": "msg_2"}})
        
        # Since "live in" is single-valued, Seattle should be overwritten
        results = memory.retrieve("Where do I live?")
        contents = [r["content"] for r in results]
        
        assert any("Portland" in text for text in contents)
        assert not any("Seattle" in text for text in contents)

    def test_conflict_resolution_multi_valued(self, memory):
        memory.add("s1", {"user_input": "I like jazz.", "metadata": {"memory_source_id": "msg_1"}})
        memory.add("s1", {"user_input": "I like Radiohead.", "metadata": {"memory_source_id": "msg_2"}})
        
        # Since "like" is multi-valued, both should be present
        results = memory.retrieve("What do I like?")
        contents = [r["content"] for r in results]
        
        assert any("jazz" in text for text in contents)
        assert any("Radiohead" in text for text in contents)
