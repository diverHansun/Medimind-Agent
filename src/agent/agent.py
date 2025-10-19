"""MediMind Agent integration.

Responsibilities:
- Compose LLM and LlamaIndex query engine
- Apply safety/guardrails and structured output
- Manage conversation context (session-level RAG)
"""

from typing import Optional, List
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM


class MediMindAgent:
    """MediMind Agent that integrates LLM and safety mechanisms."""

    def __init__(self, llm: LLM, query_engine: Optional[BaseQueryEngine] = None, safety=None):
        """Initialize the MediMindAgent.
        
        Args:
            llm: LLM instance for generating responses
            query_engine: Optional query engine for RAG
            safety: Optional safety module for guardrails
        """
        self.llm = llm
        self.query_engine = query_engine
        self.safety = safety
        self.conversation_history: List[ChatMessage] = []

    def query(self, question: str):
        """Process a question and return a response.
        
        Args:
            question: User's question
            
        Returns:
            Response from the agent
        """
        # Apply safety checks if available
        if self.safety:
            safety_check = self.safety.check(question)
            if not safety_check.is_safe:
                return safety_check.response
        
        # Add user question to conversation history
        self.conversation_history.append(
            ChatMessage(role=MessageRole.USER, content=question)
        )
        
        # If we have a query engine (RAG), use it
        if self.query_engine:
            # Use RAG to get context and generate response
            response = self.query_engine.query(question)
            answer = str(response)
        else:
            # Fallback to direct LLM completion
            response = self.llm.complete(question)
            answer = str(response)
        
        # Add assistant response to conversation history
        self.conversation_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=answer)
        )
        
        return answer
    
    def chat(self, messages: List[ChatMessage]):
        """Process a conversation and return a response.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Response from the agent
        """
        # Apply safety checks if available
        if self.safety:
            for message in messages:
                safety_check = self.safety.check(message.content)
                if not safety_check.is_safe:
                    return safety_check.response
        
        # Update conversation history
        self.conversation_history.extend(messages)
        
        # If we have a query engine (RAG), use it with chat history
        if self.query_engine:
            # For chat with RAG, we would typically use a chat engine
            # For now, we'll use the query engine with the last message
            last_message = messages[-1].content if messages else ""
            response = self.query_engine.query(last_message)
            answer = str(response)
        else:
            # Use LLM chat functionality
            response = self.llm.chat(messages)
            answer = str(response)
        
        # Add assistant response to conversation history
        self.conversation_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=answer)
        )
        
        return answer
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []