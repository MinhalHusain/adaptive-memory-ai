from src.memory.no_memory import NoMemory

def test_no_memory_retrieve():
    memory = NoMemory()
    results = memory.retrieve("test query")
    assert results == []

def test_no_memory_add():
    memory = NoMemory()
    # Should not raise any exception
    memory.add("session1", {"key": "value"})
    # Retrieve should still be empty
    assert memory.retrieve("test") == []
