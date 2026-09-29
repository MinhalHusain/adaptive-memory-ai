import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response given a list of messages."""
        pass

class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API calls."""
    
    def __init__(self, mock_response: str = "This is a mock response."):
        self.mock_response = mock_response
        self.last_messages = []
        
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        self.last_messages = messages
        return self.mock_response

class OpenAILLMClient(BaseLLMClient):
    """OpenAI implementation of the LLM client."""
    
    def __init__(self, api_key: str = None, model: str = None):
        if OpenAI is None:
            raise ImportError("openai package is not installed. Run 'pip install openai'")
            
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var.")
            
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)
        
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
