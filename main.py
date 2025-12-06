#!/usr/bin/env python3
"""Main entry point for MediMind Agent.

This script initializes the MediMind Agent with RAG capabilities.
On first run, it automatically builds the index from documents in data/documents/.
On subsequent runs, it loads the existing index from data/indexes/zhipu/.
"""

import sys
import os
import logging
import io

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src.llm.zhipu import build_llm
from src.rag.embeddings import build_embedding
from src.rag.indexing import build_index_if_not_exists
from src.rag.generation.chat_engine import build_chat_engine
from src.memory import build_chat_memory
from src.agent.agent import MediMindAgent
from src.common.config import get_index_directory, get_documents_directory


def main():
    """Main function to run the MediMind Agent with RAG."""
    print("=" * 60)
    print("MediMind Agent - Medical Health Q&A Assistant")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the conversation")
    print("-" * 60)
    
    try:
        # Initialize LLM (uses configuration from configs/llm.yaml)
        print("\n[1/4] Initializing LLM...")
        llm = build_llm()
        print("LLM initialized successfully")
        
        # Initialize embedding model
        print("\n[2/4] Initializing embedding model...")
        embed_model = build_embedding()
        print(f"Embedding model initialized (dimension: {embed_model.dimensions})")
        
        # Get directories from config
        index_dir = get_index_directory()
        documents_dir = get_documents_directory()
        
        # Build or load index
        print(f"\n[3/4] Setting up RAG index...")
        print(f"Index directory: {index_dir}")
        print(f"Documents directory: {documents_dir}")
        
        try:
            index = build_index_if_not_exists(
                documents_dir=documents_dir,
                embed_model=embed_model,
                persist_dir=index_dir,
                show_progress=True
            )

            # Build memory
            print("\n[4/4] Building chat engine with memory...")
            memory = build_chat_memory(llm=llm, token_limit=64000)

            # Build chat engine
            chat_engine = build_chat_engine(
                index=index,
                llm=llm,
                memory=memory,
                similarity_top_k=10,
                similarity_cutoff=0.1
            )

            # Initialize agent with ChatEngine (multi-turn with memory)
            agent = MediMindAgent(llm=llm, chat_engine=chat_engine)
            print("RAG mode enabled (with conversation memory)")

        except FileNotFoundError as e:
            print(f"\nWarning: {e}")
            print("Cannot initialize RAG - no documents found")
            print(f"To enable RAG, place documents in: {documents_dir}")
            print("Please add documents and restart the application")
            sys.exit(1)

        except Exception as e:
            print(f"\nFatal error during RAG initialization: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("Ready! Start asking questions...")
        print("=" * 60)
        
        # Start conversation loop
        while True:
            try:
                user_input = input("\nUser: ").strip()

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Assistant: Goodbye! Stay healthy!")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Get response from agent (with conversation memory)
                response = agent.chat(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nAssistant: Goodbye! Stay healthy!")
                break
            except EOFError:
                print("\n\nAssistant: Input stream ended. Goodbye!")
                break
            except Exception as e:
                print(f"\nAssistant: Sorry, an error occurred: {str(e)}")
                print("Please try rephrasing your question.")
    
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("1. ZHIPU_API_KEY is set in your environment or .env file")
        print("2. Configuration files are properly set up in configs/")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()