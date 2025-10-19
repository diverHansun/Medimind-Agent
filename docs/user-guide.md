# MediMind Agent User Guide

## Quick Start: Adding Documents and Using RAG

### Step 1: Prepare Your Documents

Place your medical documents in the `data/documents/` directory, organized by file type:

```
data/documents/
├── txt/
│   ├── diabetes_guide.txt
│   ├── hypertension_info.txt
│   └── medication_list.txt
├── pdf/
│   └── clinical_guidelines.pdf
├── csv/
│   └── drug_database.csv
└── md/
    └── faq.md
```

**Supported file formats:**
- `.txt` - Plain text files
- `.pdf` - PDF documents
- `.csv` - CSV data files
- `.md` - Markdown files

### Step 2: Build the Index

#### Option A: First Time (Automatic)

If you haven't built an index yet, simply run:

```bash
python main.py
```

The system will automatically:
1. Detect that no index exists
2. Load all documents from `data/documents/`
3. Generate embeddings using ZhipuAI embedding-3
4. Build a FAISS vector index
5. Save the index to `data/indexes/zhipu/`
6. Start the agent with RAG enabled

**Example output:**
```
============================================================
MediMind Agent - Medical Health Q&A Assistant
============================================================
Type 'quit' or 'exit' to end the conversation
------------------------------------------------------------

[1/4] Initializing LLM...
LLM initialized successfully

[2/4] Initializing embedding model...
Embedding model initialized (dimension: 1024)

[3/4] Setting up RAG index...
Index directory: data/indexes/zhipu
Documents directory: data/documents

Index not found. Building new index from data/documents...
Loaded 15 documents
Building index: 100%
Index built and saved to data/indexes/zhipu

[4/4] Building query engine...
RAG mode enabled

============================================================
Ready! Start asking medical questions...
============================================================
```

#### Option B: Rebuild Existing Index

If you've added new documents and need to rebuild the index:

```bash
# Method 1: Use the rebuild script
python scripts/rebuild_index.py

# Method 2: Manual deletion (Windows)
Remove-Item -Recurse -Force data/indexes/zhipu
python main.py

# Method 2: Manual deletion (Linux/Mac)
rm -rf data/indexes/zhipu
python main.py
```

### Step 3: Ask Questions

Once the agent is running, you can ask medical questions:

**Example conversation:**
```
User: What is diabetes?