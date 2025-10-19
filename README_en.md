# MediMind-Agent

MediMind-Agent is a medical health Q&A assistant using RAG (Retrieval-Augmented Generation) with ZhipuAI GLM-4, LlamaIndex, and FAISS.

## Features

- RAG-powered medical question answering
- Automatic index building on first run
- Support for multiple document formats (TXT, PDF, CSV, MD)
- Configurable retrieval parameters
- Custom postprocessors for result filtering

## Quick Start

### 1. Installation

**Option A: Using uv (Recommended)**

```bash
# Install uv if you haven't already
pip install uv

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

**Option B: Using pip**

```bash
pip install -r requirements.txt
```

### 2. Configuration

**Step 2.1: Set up API Key**

Copy the example environment file and add your API key:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your ZhipuAI API key
# ZHIPU_API_KEY=your_actual_api_key_here
```

**Step 2.2: Configure LLM and Embedding (Optional)**

All model configurations are in `configs/` directory:
- `configs/llm.yaml` - LLM parameters (model, temperature, max_tokens, etc.)
- `configs/embeddings.yaml` - Embedding model configuration
- `configs/rag.yaml` - RAG retrieval parameters

You can modify these YAML files directly or override them with environment variables in `.env`.

### 3. Prepare Documents

Place your medical documents in the `data/documents/` directory, organized by file type:

```
data/documents/
├── txt/       # Plain text files
├── pdf/       # PDF documents
├── csv/       # CSV data files
├── md/        # Markdown files
└── excel/     # Excel spreadsheets (.xlsx, .xls)
```

### 4. Run the Agent

**If using uv:**

```bash
# Make sure virtual environment is activated
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Run the agent
python main.py
```

**If using pip:**

```bash
python main.py
```

On first run, the system will:
1. Load documents from `data/documents/`
2. Build FAISS index automatically
3. Save index to `data/indexes/zhipu/`

On subsequent runs, it will load the existing index directly.

## Project Structure

```
MediMind-Agent/
├── main.py                     # Main entry point
├── src/
│   ├── agent/                  # Agent implementation
│   ├── llm/                    # LLM utilities (ZhipuAI GLM-4)
│   ├── rag/                    # RAG pipeline modules
│   │   ├── embeddings/         # Embedding models (ZhipuAI embedding-3)
│   │   ├── document_loader/    # Document loading (TXT, PDF, CSV, MD, Excel)
│   │   ├── text_splitter/      # Text chunking
│   │   ├── indexing/           # Index building and loading
│   │   ├── generation/         # Query engine
│   │   └── postprocessors/     # Result filtering
│   └── common/                 # Common utilities and config
├── configs/                    # Configuration files
│   ├── embeddings.yaml         # Embedding model config
│   └── rag.yaml               # RAG parameters
├── data/                       # Data storage
│   ├── documents/              # Raw documents (user-provided)
│   └── indexes/                # Persistent indexes
└── docs/                       # Documentation
    ├── architecture.md         # Architecture overview
    ├── rag-workflow.md         # RAG workflow details
    ├── storage.md              # Storage design
    └── api-usage.md            # API documentation
```

## Configuration

### Configuration Strategy

**Primary Configuration: YAML Files (Recommended)**

All model and system configurations are managed in `configs/` directory:

**LLM Configuration (`configs/llm.yaml`)**

```yaml
llm:
  zhipu:
    model: glm-4
    temperature: 0.2
    max_tokens: 2048
    top_p: 0.7
    system_prompt: "You are a professional medical assistant..."
```

**Environment Variables: API Keys Only**

The `.env` file should only contain sensitive information like API keys:

```bash
# .env (DO NOT commit to git)
ZHIPU_API_KEY=your_actual_api_key_here
```

**Optional Overrides**

You can override YAML configs with environment variables if needed:
```bash
# In .env or shell
LLM_MODEL=glm-4.5
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=4096
```

**Configuration Priority:**
1. Function parameters (highest)
2. Environment variables (.env)
3. YAML config files (configs/)
4. Code defaults (lowest)

### Embedding Model (`configs/embeddings.yaml`)

```yaml
embeddings:
  zhipu:
    enabled: true
    model: embedding-3
    dim: 1024
    index_dir: data/indexes/zhipu
```

### RAG Parameters (`configs/rag.yaml`)

```yaml
rag:
  mode: auto
  top_k:
    default: 5
  min_relevance: 0.25
  text_splitter:
    chunk_size: 512
    chunk_overlap: 50
```

## Usage Examples

### Basic Query

```python
from src.llm.zhipu import build_llm
from src.agent.agent import MediMindAgent

llm = build_llm()
agent = MediMindAgent(llm=llm)
response = agent.query("What is diabetes?")
```

### With RAG

```python
from src.llm.zhipu import build_llm
from src.rag import build_embedding, build_index_if_not_exists, build_query_engine

llm = build_llm()
embed_model = build_embedding()

index = build_index_if_not_exists(
    documents_dir="data/documents",
    embed_model=embed_model,
    persist_dir="data/indexes/zhipu"
)

query_engine = build_query_engine(index, llm=llm, top_k=5)
agent = MediMindAgent(llm=llm, query_engine=query_engine)

response = agent.query("What are the treatment options for diabetes?")
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [RAG Workflow](docs/rag-workflow.md)
- [Storage Design](docs/storage.md)
- [API Usage](docs/api-usage.md)
- [Technical Plan](medimind-guide.md)

## Requirements

- Python 3.10+
- ZhipuAI API key
- Dependencies listed in `requirements.txt`

## License

See LICENSE file for details.
