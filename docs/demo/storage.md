# Storage Architecture Documentation

## Overview

This document describes the data storage architecture for MediMind Agent, including the organization of raw documents, index persistence, and the integration with LlamaIndex's storage mechanisms.

## Directory Structure

```
data/
├── documents/              # Raw document storage (user-provided)
│   ├── txt/               # Plain text files
│   ├── pdf/               # PDF documents
│   ├── csv/               # CSV data files
│   └── md/                # Markdown files (optional)
│
└── indexes/               # Persistent index storage
    └── zhipu/             # ZhipuAI embedding-3 index
        ├── docstore.json           # Document store
        ├── index_store.json        # Index metadata
        ├── vector_store.json       # Vector store configuration
        └── default__vector_store.json  # FAISS index data
```

## Data Categories

### 1. Raw Documents (`data/documents/`)

Raw documents are organized by file type rather than content category. This approach:
- Simplifies file management
- Aligns with LlamaIndex's `SimpleDirectoryReader` capabilities
- Allows flexible content organization by users

**Supported File Types:**
- `.txt` - Plain text medical documents
- `.pdf` - Medical guidelines, research papers
- `.csv` - Structured medical data
- `.md` - Markdown formatted documents

**User Responsibilities:**
- Place documents in appropriate subdirectories by file type
- Ensure documents are UTF-8 encoded (for text files)
- No specific naming conventions required

### 2. Index Storage (`data/indexes/zhipu/`)

Index storage contains LlamaIndex's persistent index files. These files are automatically generated and managed by the system.

**LlamaIndex Persistence Components:**

1. **docstore.json**
   - Stores the actual document content and metadata
   - Maps document IDs to document objects
   - Includes text content and custom metadata

2. **index_store.json**
   - Stores index structure and relationships
   - Maps index IDs to their configurations
   - Maintains index-to-document mappings

3. **vector_store.json**
   - Stores vector store configuration
   - Contains FAISS index parameters
   - Links to the actual vector data file

4. **default__vector_store.json**
   - Contains the actual FAISS vector index data
   - Binary format storing document embeddings
   - Used for similarity search operations

## Index Lifecycle

### First Run (Auto-Build)

When the application starts for the first time:

1. **Check Index Existence**
   - System checks for index files in `data/indexes/zhipu/`
   - Specifically looks for `docstore.json` and `vector_store.json`

2. **Auto-Build Trigger**
   - If index files are missing, auto-build is triggered
   - System scans `data/documents/` for all supported file types
   - Progress indicators show indexing status

3. **Index Construction**
   - Documents are loaded using `SimpleDirectoryReader`
   - Text is split into chunks (default: 512 tokens, 50 overlap)
   - Embeddings are generated using ZhipuAI embedding-3
   - FAISS index is built with inner product similarity
   - All components are persisted to `data/indexes/zhipu/`

4. **Completion**
   - Index is ready for retrieval operations
   - System proceeds to normal operation mode

### Subsequent Runs

On subsequent application starts:

1. **Load Existing Index**
   - System detects existing index files
   - Loads index using `FaissVectorStore.from_persist_dir()`
   - Reconstructs `VectorStoreIndex` from persisted data

2. **No Rebuild**
   - Existing index is used as-is
   - No document scanning or re-indexing occurs
   - Fast startup time

### Index Updates

**Current Strategy: Manual Rebuild**

To update the index with new documents:

1. Delete existing index directory: `data/indexes/zhipu/`
2. Add new documents to `data/documents/`
3. Restart application to trigger auto-build

**Future Enhancements:**
- Incremental index updates
- Document change detection
- Selective re-indexing

## Configuration Integration

### Embedding Configuration (`configs/embeddings.yaml`)

```yaml
embeddings:
  zhipu:
    enabled: true
    provider: zhipuai
    model: embedding-3
    dim: 1024
    index_dir: data/indexes/zhipu  # Index storage location
```

**Key Parameters:**
- `index_dir`: Specifies where to persist/load the index
- `dim`: Must match embedding-3 model dimension (1024)
- `model`: Must be a valid ZhipuAI embedding model

### Environment Variables (`.env`)

```bash
ZHIPU_API_KEY=your_api_key_here
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSION=1024
```

**Priority Order:**
1. `configs/embeddings.yaml` (primary)
2. Environment variables (override)

## Storage Best Practices

### Document Preparation

1. **File Organization**
   - Place files in correct subdirectory by type
   - Use descriptive filenames
   - Avoid special characters in filenames

2. **Content Quality**
   - Ensure text is readable and well-formatted
   - Remove unnecessary headers/footers from PDFs
   - Validate CSV structure before adding

3. **Size Considerations**
   - Individual files: Recommended < 10MB
   - Total corpus: No hard limit, but affects build time
   - Large corpora may require increased memory

### Index Management

1. **Backup**
   - Regularly backup `data/indexes/` directory
   - Backup is faster than rebuilding from scratch
   - Consider version control for index snapshots

2. **Disk Space**
   - Index size roughly 1.5-2x the size of raw documents
   - FAISS index grows linearly with document count
   - Monitor available disk space

3. **Performance**
   - Larger indexes may have slower retrieval
   - Consider splitting very large corpora
   - Use SSD storage for better performance

## Technical Details

### FAISS Index Configuration

- **Index Type**: `IndexFlatIP` (Inner Product)
- **Similarity Metric**: Cosine similarity (via normalized vectors)
- **Dimension**: 1024 (matching embedding-3)
- **Storage Format**: Binary serialization

### LlamaIndex Storage Context

The system uses LlamaIndex's `StorageContext` to manage persistence:

```python
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir=persist_dir
)
```

This ensures all index components are properly synchronized and persisted together.

## Troubleshooting

### Index Build Failures

**Problem**: Index build fails during first run

**Possible Causes:**
- Missing `ZHIPU_API_KEY` in environment
- No documents in `data/documents/`
- Insufficient disk space
- Network issues accessing ZhipuAI API

**Solutions:**
- Verify API key is set correctly
- Check document directory contains files
- Ensure adequate disk space (2x document size)
- Check network connectivity

### Index Load Failures

**Problem**: Cannot load existing index

**Possible Causes:**
- Corrupted index files
- Mismatched embedding dimensions
- Incomplete index (interrupted build)

**Solutions:**
- Delete index directory and rebuild
- Verify configuration matches original build
- Ensure all index files are present

### Performance Issues

**Problem**: Slow retrieval or high memory usage

**Possible Causes:**
- Very large index size
- Insufficient RAM
- Disk I/O bottleneck

**Solutions:**
- Consider corpus splitting
- Increase available RAM
- Use SSD storage
- Optimize chunk size parameters

## Future Enhancements

### Planned Features

1. **Incremental Updates**
   - Add new documents without full rebuild
   - Track document changes and update selectively
   - Maintain index version history

2. **Multi-Index Support**
   - Separate indexes for different document types
   - Domain-specific indexes (e.g., guidelines vs. FAQs)
   - Query routing based on content type

3. **Index Optimization**
   - Periodic index compaction
   - Remove outdated or unused documents
   - Optimize FAISS index structure for speed

4. **Monitoring and Analytics**
   - Index size and growth tracking
   - Query performance metrics
   - Document coverage analysis

## References

- [LlamaIndex Storage Documentation](https://docs.llamaindex.ai/en/stable/module_guides/storing/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [ZhipuAI Embedding API](https://open.bigmodel.cn/dev/api#text_embedding)

