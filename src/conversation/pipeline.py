from typing import Any, Dict, List, Optional
from src.llm.client import BaseLLMClient
from src.memory.base import BaseMemory
from src.conversation.session import ConversationSession
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ConversationalPipeline:
    """Coordinates LLM, session history, and persistent memory."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        memory: BaseMemory,
        system_prompt: str = "You are a helpful AI assistant."
    ):
        self.llm_client = llm_client
        self.memory = memory
        self.system_prompt = system_prompt
        self.last_retrieved_memories: List[Dict[str, Any]] = []
        logger.info(f"Initialized ConversationalPipeline with memory: {type(self.memory).__name__}")

    def process_message(
        self, text: str, session: ConversationSession, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a user message and generate a response."""
        logger.info(f"Processing message in session: {session.session_id}")

        # 1. Retrieve relevant memories (returns empty for NoMemory baseline)
        retrieved_memories = self.memory.retrieve(query=text, session_id=session.session_id)
        self.last_retrieved_memories = retrieved_memories

        # 2. Add user message to session
        session.add_user_message(text)

        # 3. Construct messages for LLM
        messages = [{"role": "system", "content": self.system_prompt}]

        if retrieved_memories:
            # Format retrieved memories as clean text for the LLM context
            memory_lines = []
            for m in retrieved_memories:
                content = m.get("content", str(m))
                memory_lines.append(f"- {content}")
            memory_context = "\n".join(memory_lines)
            messages.append({
                "role": "system",
                "content": f"Relevant information from previous conversations:\n{memory_context}"
            })

        # Append session history
        messages.extend(session.get_history())

        # 4. Generate response
        try:
            response_text = self.llm_client.generate_response(messages)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response_text = "I encountered an error processing your request."

        # 5. Add assistant response to session
        session.add_assistant_message(response_text)

        # 6. Store in memory (does nothing for NoMemory baseline)
        memory_data = {"user_input": text, "assistant_response": response_text}
        if metadata:
            memory_data["metadata"] = metadata
            
        self.memory.add(session_id=session.session_id, data=memory_data)

        return response_text
