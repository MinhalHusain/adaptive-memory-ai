import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversation.pipeline import ConversationalPipeline
from src.conversation.session import ConversationSession
from src.memory.no_memory import NoMemory
from src.llm.client import MockLLMClient, OpenAILLMClient

def main():
    print("=== Running No-Memory Baseline Experiment ===")
    
    # Load .env file
    load_dotenv()
    
    # Check if OPENAI_API_KEY is available, otherwise use Mock
    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAILLMClient")
        try:
            llm = OpenAILLMClient()
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}. Falling back to MockLLMClient.")
            llm = MockLLMClient(mock_response="[Mock Response] I don't remember that.")
    else:
        print("OPENAI_API_KEY not found. Using MockLLMClient.")
        llm = MockLLMClient(mock_response="[Mock Response] This is a baseline test.")
        
    memory = NoMemory()
    pipeline = ConversationalPipeline(llm_client=llm, memory=memory)
    
    print("\n--- Session 1 ---")
    session1 = ConversationSession("session-1")
    msg1 = "Hi, my name is Alex and I like apples."
    print(f"User (s1): {msg1}")
    resp1 = pipeline.process_message(msg1, session1)
    print(f"AI (s1): {resp1}")
    
    print("\n--- Session 2 ---")
    session2 = ConversationSession("session-2")
    msg2 = "What is my name, and what fruit do I like?"
    print(f"User (s2): {msg2}")
    resp2 = pipeline.process_message(msg2, session2)
    print(f"AI (s2): {resp2}")
    
    print("\nBaseline test completed.")
    print("Notice how Session 2 does not have access to Session 1's information.")

if __name__ == "__main__":
    main()
