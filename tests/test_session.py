from src.conversation.session import ConversationSession

def test_session_history_isolation():
    session1 = ConversationSession("session1")
    session2 = ConversationSession("session2")
    
    session1.add_user_message("Hello from 1")
    session2.add_user_message("Hello from 2")
    
    assert len(session1.get_history()) == 1
    assert session1.get_history()[0]["content"] == "Hello from 1"
    
    assert len(session2.get_history()) == 1
    assert session2.get_history()[0]["content"] == "Hello from 2"
    
def test_session_clear():
    session = ConversationSession()
    session.add_user_message("test")
    assert len(session.get_history()) == 1
    session.clear()
    assert len(session.get_history()) == 0
