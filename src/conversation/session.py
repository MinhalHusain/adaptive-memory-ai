import uuid
from typing import Dict, List

class ConversationSession:
    """Manages short-term conversation history for a single session."""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.history: List[Dict[str, str]] = []
        
    def add_user_message(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        
    def add_assistant_message(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        
    def get_history(self) -> List[Dict[str, str]]:
        return self.history
        
    def clear(self) -> None:
        self.history = []
