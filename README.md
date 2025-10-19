# MediMind-Agent

MediMind-Agent 是一个基于 RAG（检索增强生成）技术的医疗健康问答助手，使用 ZhipuAI GLM-4、LlamaIndex 和 FAISS。

## 功能特性

- 基于 RAG 的医疗问题回答
- 首次运行时自动构建索引
- 支持多种文档格式（TXT、PDF、CSV、MD）
- 可配置的检索参数
- 自定义后处理器用于结果过滤

## 快速开始

### 1. 安装

**方式一：使用 uv（推荐）**

```bash
# 如果还没有安装 uv，请先安装
pip install uv

# 安装依赖并创建虚拟环境
uv sync

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**方式二：使用 pip**

```bash
pip install -r requirements.txt
```

### 2. 配置

**步骤 2.1：设置 API 密钥**

复制示例环境文件并添加您的 API 密钥：

```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件，添加您的 ZhipuAI API 密钥
# ZHIPU_API_KEY=your_actual_api_key_here
```

**步骤 2.2：配置 LLM 和嵌入模型（可选）**

所有模型配置都在 `configs/` 目录中：
- `configs/llm.yaml` - LLM 参数（模型、温度、最大令牌数等）
- `configs/embeddings.yaml` - 嵌入模型配置
- `configs/rag.yaml` - RAG 检索参数

您可以直接修改这些 YAML 文件，或在 `.env` 中使用环境变量覆盖它们。

### 3. 准备文档

将您的医疗文档放置在 `data/documents/` 目录中，按文件类型组织：

```
data/documents/
├── txt/       # 纯文本文件
├── pdf/       # PDF 文档
├── csv/       # CSV 数据文件
├── md/        # Markdown 文件
└── excel/     # Excel 电子表格（.xlsx、.xls）
```

### 4. 运行代理

**如果使用 uv：**

```bash
# 确保虚拟环境已激活
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 运行代理
python main.py
```

**如果使用 pip：**

```bash
python main.py
```

首次运行时，系统将：
1. 从 `data/documents/` 加载文档
2. 自动构建 FAISS 索引
3. 将索引保存到 `data/indexes/zhipu/`

后续运行时，将直接加载现有索引。

## 项目结构

```
MediMind-Agent/
├── main.py                     # 主入口点
├── src/
│   ├── agent/                  # 代理实现
│   ├── llm/                    # LLM 工具（ZhipuAI GLM-4）
│   ├── rag/                    # RAG 管道模块
│   │   ├── embeddings/         # 嵌入模型（ZhipuAI embedding-3）
│   │   ├── document_loader/    # 文档加载（TXT、PDF、CSV、MD、Excel）
│   │   ├── text_splitter/      # 文本分块
│   │   ├── indexing/           # 索引构建和加载
│   │   ├── generation/         # 查询引擎
│   │   └── postprocessors/     # 结果过滤
│   └── common/                 # 通用工具和配置
├── configs/                    # 配置文件
│   ├── embeddings.yaml         # 嵌入模型配置
│   └── rag.yaml               # RAG 参数
├── data/                       # 数据存储
│   ├── documents/              # 原始文档（用户提供）
│   └── indexes/                # 持久化索引
└── docs/                       # 文档
    ├── architecture.md         # 架构概述
    ├── rag-workflow.md         # RAG 工作流详情
    ├── storage.md              # 存储设计
    └── api-usage.md            # API 文档
```

## 配置

### 配置策略

**主要配置：YAML 文件（推荐）**

所有模型和系统配置都在 `configs/` 目录中管理：

**LLM 配置（`configs/llm.yaml`）**

```yaml
llm:
  zhipu:
    model: glm-4
    temperature: 0.2
    max_tokens: 2048
    top_p: 0.7
    system_prompt: "You are a professional medical assistant..."
```

**环境变量：仅 API 密钥**

`.env` 文件应仅包含敏感信息，如 API 密钥：

```bash
# .env（不要提交到 git）
ZHIPU_API_KEY=your_actual_api_key_here
```

**可选覆盖**

如果需要，您可以使用环境变量覆盖 YAML 配置：
```bash
# 在 .env 或 shell 中
LLM_MODEL=glm-4.5
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=4096
```

**配置优先级：**
1. 函数参数（最高）
2. 环境变量（.env）
3. YAML 配置文件（configs/）
4. 代码默认值（最低）

### 嵌入模型（`configs/embeddings.yaml`）

```yaml
embeddings:
  zhipu:
    enabled: true
    model: embedding-3
    dim: 1024
    index_dir: data/indexes/zhipu
```

### RAG 参数（`configs/rag.yaml`）

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

## 使用示例

### 基本查询

```python
from src.llm.zhipu import build_llm
from src.agent.agent import MediMindAgent

llm = build_llm()
agent = MediMindAgent(llm=llm)
response = agent.query("什么是糖尿病？")
```

### 使用 RAG

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

response = agent.query("糖尿病的治疗方案有哪些？")
```

## 文档

- [架构概述](docs/architecture.md)
- [RAG 工作流](docs/rag-workflow.md)
- [存储设计](docs/storage.md)
- [API 使用](docs/api-usage.md)
- [技术计划](medimind-guide.md)

## 系统要求

- Python 3.10+
- ZhipuAI API 密钥
- `requirements.txt` 中列出的依赖项

## 许可证

详见 LICENSE 文件。
