from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseMemory(ABC):
    """Abstract interface for memory management."""
    
    @abstractmethod
    def add(self, session_id: str, data: Dict[str, Any]) -> None:
        """Add new information to memory."""
        pass
        
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant information from memory based on a query."""
        pass
        
    @abstractmethod
    def update(self, memory_id: str, data: Dict[str, Any]) -> None:
        """Update an existing memory."""
        pass
        
    @abstractmethod
    def forget(self, memory_id: str) -> None:
        """Delete/forget a specific memory."""
        pass

    def reset(self) -> None:
        """Reset/clear all memory state. Override in subclasses that
        maintain persistent storage."""
        pass
