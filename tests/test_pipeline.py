from src.conversation.pipeline import ConversationalPipeline
from src.conversation.session import ConversationSession
from src.memory.no_memory import NoMemory
from src.llm.client import MockLLMClient

def test_pipeline_basic_interaction():
    llm = MockLLMClient(mock_response="Mock response")
    memory = NoMemory()
    pipeline = ConversationalPipeline(llm_client=llm, memory=memory)
    
    session = ConversationSession("test-session")
    
    response = pipeline.process_message("Hi there", session)
    
    assert response == "Mock response"
    assert len(session.get_history()) == 2
    assert session.get_history()[0]["role"] == "user"
    assert session.get_history()[1]["role"] == "assistant"
    assert session.get_history()[1]["content"] == "Mock response"
    
def test_pipeline_no_memory_leak():
    llm = MockLLMClient(mock_response="Mock response")
    memory = NoMemory()
    pipeline = ConversationalPipeline(llm_client=llm, memory=memory)
    
    session1 = ConversationSession("s1")
    session2 = ConversationSession("s2")
    
    pipeline.process_message("Message in session 1", session1)
    
    # In a real memory system, session2 might retrieve info from session1.
    # Here, memory retrieve should return empty.
    # We can inspect the LLM client's last_messages to see if it got anything.
    pipeline.process_message("Message in session 2", session2)
    
    messages_sent = llm.last_messages
    # Should contain system prompt and user message for session 2
    assert len(messages_sent) == 2
    assert messages_sent[1]["content"] == "Message in session 2"
