"""Script to rebuild the RAG index from scratch.

This script deletes the existing index and rebuilds it from all documents
in the data/documents/ directory.

Usage:
    python scripts/rebuild_index.py
"""

import sys
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.embeddings import build_embedding
from src.rag.indexing import build_index
from src.rag.document_loader import load_documents_from_subdirs
from src.common.config import get_index_directory, get_documents_directory


def rebuild_index():
    """Rebuild the RAG index from scratch."""
    
    # Get directories
    index_dir = get_index_directory()
    documents_dir = get_documents_directory()
    
    print("=" * 60)
    print("RAG Index Rebuild Tool")
    print("=" * 60)
    
    # Check if documents directory exists
    if not os.path.exists(documents_dir):
        print(f"\nError: Documents directory not found: {documents_dir}")
        print("Please create the directory and add documents first.")
        return
    
    # Delete existing index
    if os.path.exists(index_dir):
        print(f"\n[1/4] Deleting existing index at {index_dir}...")
        shutil.rmtree(index_dir)
        print("Existing index deleted")
    else:
        print(f"\n[1/4] No existing index found at {index_dir}")
    
    # Initialize embedding model
    print("\n[2/4] Initializing embedding model...")
    try:
        embed_model = build_embedding()
        print(f"Embedding model initialized (dimension: {embed_model.dimensions})")
    except Exception as e:
        print(f"\nError: Failed to initialize embedding model: {e}")
        print("Please check your ZHIPU_API_KEY is set correctly.")
        return
    
    # Load documents from organized subdirectories
    print(f"\n[3/4] Loading documents from {documents_dir}...")
    try:
        documents = load_documents_from_subdirs(documents_dir, recursive=False)
        print(f"Loaded {len(documents)} documents")
        
        # Show document details
        print("\nDocuments loaded:")
        for i, doc in enumerate(documents[:5], 1):  # Show first 5
            file_path = doc.metadata.get('file_path', 'Unknown')
            file_name = doc.metadata.get('file_name', 'Unknown')
            print(f"  {i}. {file_name} ({len(doc.text)} chars)")
        
        if len(documents) > 5:
            print(f"  ... and {len(documents) - 5} more documents")
            
    except Exception as e:
        print(f"\nError: Failed to load documents: {e}")
        return
    
    # Build index
    print(f"\n[4/4] Building index and saving to {index_dir}...")
    try:
        index = build_index(
            documents=documents,
            embed_model=embed_model,
            persist_dir=index_dir,
            show_progress=True
        )
        print(f"\nIndex built successfully!")
        print(f"Index saved to: {index_dir}")
        
        # Verify index files
        print("\nIndex files created:")
        for file in os.listdir(index_dir):
            file_path = os.path.join(index_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size} bytes)")
        
    except Exception as e:
        print(f"\nError: Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Index rebuild completed successfully!")
    print("=" * 60)
    print("\nYou can now run 'python main.py' to use the updated index.")


if __name__ == "__main__":
    rebuild_index()

