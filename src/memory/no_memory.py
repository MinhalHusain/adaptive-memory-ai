from typing import Any, Dict, List
from src.memory.base import BaseMemory

class NoMemory(BaseMemory):
    """Baseline implementation that stores and retrieves nothing."""
    
    def add(self, session_id: str, data: Dict[str, Any]) -> None:
        # Do nothing
        pass
        
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        # Always return empty result
        return []
        
    def update(self, memory_id: str, data: Dict[str, Any]) -> None:
        # Do nothing
        pass
        
    def forget(self, memory_id: str) -> None:
        # Do nothing
        pass
