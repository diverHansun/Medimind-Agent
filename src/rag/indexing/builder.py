"""Index building and loading utilities using FAISS + LlamaIndex.

This module provides functions to build and manage FAISS vector indexes
using LlamaIndex framework. It supports creating indexes from documents,
persisting them to disk, and loading them back.
"""

import os
import json
import faiss
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.embeddings import BaseEmbedding


def save_index_metadata(persist_dir: str, embed_model: BaseEmbedding) -> None:
    """Save index metadata including embedding dimension and model info.

    Metadata is used to detect dimension mismatches when loading indexes.
    Follows KISS principle: simple JSON file for metadata storage.

    Args:
        persist_dir: Directory where index is persisted
        embed_model: Embedding model used to build the index
    """
    # Get model name safely (handle property objects)
    try:
        model_name = embed_model.model_name
    except (AttributeError, TypeError):
        model_name = "unknown"

    metadata = {
        "embedding_dimension": embed_model.dimensions,
        "model_name": str(model_name),
    }

    metadata_path = os.path.join(persist_dir, "index_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_index_metadata(persist_dir: str) -> dict:
    """Load index metadata from persist directory.

    Args:
        persist_dir: Directory where index is persisted

    Returns:
        dict: Metadata dictionary, or empty dict if not found
    """
    metadata_path = os.path.join(persist_dir, "index_metadata.json")
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_dimension_compatibility(
    persist_dir: str,
    embed_model: BaseEmbedding
) -> tuple[bool, str]:
    """Check if embedding model dimension matches persisted index.

    Implements automatic dimension detection for index rebuild.
    Follows Single Responsibility Principle (SRP).

    Args:
        persist_dir: Directory where index is persisted
        embed_model: Embedding model to check

    Returns:
        tuple: (is_compatible, message)
            - is_compatible: True if dimensions match, False otherwise
            - message: Explanation message
    """
    metadata = load_index_metadata(persist_dir)

    # If no metadata, assume compatible (legacy index)
    if not metadata:
        return True, "No metadata found (legacy index), assuming compatible"

    stored_dim = metadata.get("embedding_dimension")
    current_dim = embed_model.dimensions
    stored_model = metadata.get("model_name", "unknown")
    current_model = getattr(embed_model, "model_name", "unknown")

    if stored_dim != current_dim:
        message = (
            f"Dimension mismatch detected!\n"
            f"  Stored index: {stored_model} ({stored_dim}D)\n"
            f"  Current model: {current_model} ({current_dim}D)\n"
            f"  Index will be rebuilt automatically."
        )
        return False, message

    return True, f"Dimensions match ({current_dim}D)"


def build_index(
    documents: List[Document],
    embed_model: BaseEmbedding,
    persist_dir: Optional[str] = None,
    show_progress: bool = True,
) -> VectorStoreIndex:
    """Build a FAISS vector index from documents using LlamaIndex.

    The embedding dimension is automatically obtained from the embed_model,
    ensuring consistency between the model and FAISS index.

    Args:
        documents: List of Document objects to index
        embed_model: Embedding model to use for vectorization
        persist_dir: Directory to persist the index (optional)
        show_progress: Whether to show progress bar during indexing

    Returns:
        VectorStoreIndex: The created index

    Example:
        >>> from llama_index.core import Document
        >>> from src.embeddings.zhipu import build_embedding
        >>>
        >>> documents = [Document(text="Diabetes is a chronic disease...")]
        >>> embed_model = build_embedding()
        >>> index = build_index(documents, embed_model, persist_dir="data/indexes/zhipu")
    """
    # Auto-get dimension from embedding model
    dim = embed_model.dimensions

    # Create FAISS index using inner product for cosine similarity
    faiss_index = faiss.IndexFlatIP(dim)

    # Create vector store
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index from documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=show_progress
    )

    # Persist index if directory is provided
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        # Save metadata for dimension checking
        save_index_metadata(persist_dir, embed_model)

    return index


def load_index(
    persist_dir: str,
    embed_model: BaseEmbedding,
) -> VectorStoreIndex:
    """Load a FAISS vector index from disk.
    
    Args:
        persist_dir: Directory where the index is stored
        embed_model: Embedding model used for the index
        
    Returns:
        VectorStoreIndex: The loaded index
        
    Raises:
        FileNotFoundError: If index files are not found in persist_dir
        
    Example:
        >>> from src.embeddings.zhipu import build_embedding
        >>> 
        >>> embed_model = build_embedding()
        >>> index = load_index("data/indexes/zhipu", embed_model)
    """
    if not index_exists(persist_dir):
        raise FileNotFoundError(
            f"Index not found in {persist_dir}. "
            "Please build the index first or check the directory path."
        )
    
    # Load vector store from persisted directory
    vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, 
        persist_dir=persist_dir
    )
    
    # Load index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    return index


def index_exists(persist_dir: str) -> bool:
    """Check if a valid index exists in the specified directory.
    
    Args:
        persist_dir: Directory to check for index files
        
    Returns:
        bool: True if index exists, False otherwise
        
    Example:
        >>> if index_exists("data/indexes/zhipu"):
        ...     print("Index found")
        ... else:
        ...     print("Index not found, will build")
    """
    if not os.path.exists(persist_dir):
        return False
    
    # Check for required index files
    required_files = ["docstore.json", "vector_store.json"]
    for filename in required_files:
        filepath = os.path.join(persist_dir, filename)
        if not os.path.exists(filepath):
            return False
    
    return True


def build_index_if_not_exists(
    documents_dir: str,
    embed_model: BaseEmbedding,
    persist_dir: str,
    show_progress: bool = True,
    force_rebuild: bool = False,
) -> VectorStoreIndex:
    """Build index if it doesn't exist or needs rebuild, otherwise load existing index.

    This function implements intelligent auto-build behavior:
    1. If index doesn't exist: build from documents
    2. If index exists but dimension mismatches: rebuild automatically
    3. If index exists and compatible: load existing index
    4. If force_rebuild=True: always rebuild

    Follows DRY and KISS principles with automatic dimension detection.

    Args:
        documents_dir: Directory containing documents to index
        embed_model: Embedding model to use
        persist_dir: Directory to persist/load the index
        show_progress: Whether to show progress during building
        force_rebuild: Force rebuild even if compatible index exists

    Returns:
        VectorStoreIndex: Either newly built or loaded index

    Raises:
        FileNotFoundError: If documents_dir doesn't exist and index doesn't exist
        ValueError: If no documents found in documents_dir

    Example:
        >>> from src.embeddings.zhipu import build_embedding
        >>>
        >>> embed_model = build_embedding()
        >>> index = build_index_if_not_exists(
        ...     documents_dir="data/documents",
        ...     embed_model=embed_model,
        ...     persist_dir="data/indexes/zhipu"
        ... )
    """
    # Check if force rebuild requested
    if force_rebuild and index_exists(persist_dir):
        print(f"[Index] Force rebuild requested, rebuilding index...")
        needs_rebuild = True
    # Check if index exists
    elif index_exists(persist_dir):
        # Check dimension compatibility
        is_compatible, message = check_dimension_compatibility(persist_dir, embed_model)
        print(f"[Index] {message}")

        if is_compatible:
            # Load existing compatible index
            print(f"[Index] Loading existing index from {persist_dir}...")
            return load_index(persist_dir, embed_model)
        else:
            # Dimension mismatch, need rebuild
            print(f"[Index] Rebuilding index due to dimension mismatch...")
            needs_rebuild = True
    else:
        # Index doesn't exist
        print(f"[Index] No index found, building new index from {documents_dir}...")
        needs_rebuild = True

    # Build new index
    if needs_rebuild or not index_exists(persist_dir):
        # Load documents from organized subdirectories (txt/, pdf/, csv/, md/, excel/)
        from src.rag.document_loader import load_documents_from_subdirs
        print("[Index] Loading documents from subdirectories...")
        documents = load_documents_from_subdirs(documents_dir, recursive=False)

        print(f"[Index] Successfully loaded {len(documents)} documents")

        # Build and persist index
        index = build_index(
            documents=documents,
            embed_model=embed_model,
            persist_dir=persist_dir,
            show_progress=show_progress
        )

        print(f"[Index] Index built and saved to {persist_dir}")

        return index


def add_nodes_to_index(
    index: VectorStoreIndex,
    nodes: List[BaseNode]
) -> None:
    """Add nodes to an existing index.
    
    Args:
        index: The existing index to add nodes to
        nodes: List of nodes to add
        
    Example:
        >>> from src.rag.text_splitter import split_documents
        >>> nodes = split_documents(new_documents)
        >>> add_nodes_to_index(index, nodes)
    """
    index.insert_nodes(nodes)


def add_documents_to_index(
    index: VectorStoreIndex,
    documents: List[Document]
) -> None:
    """Add documents to an existing index.
    
    Args:
        index: The existing index to add documents to
        documents: List of documents to add
        
    Example:
        >>> new_docs = [Document(text="New medical information...")]
        >>> add_documents_to_index(index, new_docs)
    """
    for doc in documents:
        index.insert(doc)


