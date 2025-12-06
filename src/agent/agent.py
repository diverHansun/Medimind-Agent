"""MediMind Agent integration.

Responsibilities:
- Integrate chat engine with memory management
- Apply safety/guardrails if available
- Coordinate multi-turn conversations
"""

from typing import Optional
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms import LLM


class MediMindAgent:
    """MediMind Agent that integrates ChatEngine with memory management."""

    def __init__(self, llm: LLM, chat_engine: BaseChatEngine, safety=None):
        """Initialize the MediMindAgent.

        Args:
            llm: LLM instance for generating responses
            chat_engine: ChatEngine instance for multi-turn conversations with memory
            safety: Optional safety module for guardrails
        """
        self.llm = llm
        self.chat_engine = chat_engine
        self.safety = safety

    def chat(self, message: str) -> str:
        """Chat with the agent using multi-turn conversation memory.

        The ChatEngine automatically handles:
        - Retrieving conversation history from memory
        - Condensing history with new message to form standalone question
        - Retrieving relevant documents
        - Generating response with context and history
        - Saving conversation to memory

        Args:
            message: User's message

        Returns:
            str: Response from the agent

        Example:
            >>> agent = MediMindAgent(llm, chat_engine)
            >>> response = agent.chat("What is diabetes?")
            >>> print(response)
            >>> response = agent.chat("What are its symptoms?")  # Has context from previous message
            >>> print(response)
        """
        # Apply safety checks if available
        if self.safety:
            safety_check = self.safety.check(message)
            if not safety_check.is_safe:
                return safety_check.response

        try:
            # Chat through engine (memory management is automatic)
            response = self.chat_engine.chat(message)
            response_text = str(response).strip()

            # Handle empty response
            if not response_text or response_text == "Empty Response":
                # Fallback: query LLM directly when no relevant documents found
                return self._fallback_response(message)

            return response_text

        except Exception as e:
            # Log error and provide user-friendly response
            print(f"[Error during chat] {str(e)}")
            return f"I encountered an error while processing your request. Please try again or rephrase your question."

    def _fallback_response(self, message: str) -> str:
        """Generate response directly from LLM when no documents are found.

        This is used when ChatEngine returns empty response due to:
        - No relevant documents in knowledge base
        - Query not matching any indexed content
        - All documents filtered out by similarity threshold

        Args:
            message: User's message

        Returns:
            str: Response from LLM without RAG context
        """
        from llama_index.core.llms import ChatMessage, MessageRole

        try:
            # Create a simple prompt for direct LLM response
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful AI assistant. Provide helpful, accurate, and concise responses."
                ),
                ChatMessage(role=MessageRole.USER, content=message)
            ]

            response = self.llm.chat(messages)
            return str(response.message.content).strip()

        except Exception as e:
            print(f"[Error in fallback response] {str(e)}")
            return "I apologize, but I'm unable to process your request at the moment. Please try again later."

    def reset_conversation(self) -> None:
        """Reset the conversation memory."""
        self.chat_engine.reset()