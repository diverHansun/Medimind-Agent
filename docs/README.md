# MediMind Agent 文档中心

欢迎来到 MediMind Agent 文档中心。这里包含了项目的完整技术文档，帮助您快速了解和使用这个基于 LlamaIndex 的医学问答系统。

## 文档结构

### 1. [项目架构文档](./architecture.md)
- **开发目的**：详细说明 MediMind Agent 的设计目标和核心功能
- **架构选择**：解释为什么选择 LlamaIndex、BioBERT、智谱AI 等技术栈
- **系统架构**：完整的系统架构图和模块职责划分
- **技术栈**：核心框架、嵌入模型、大语言模型的选择理由
- **设计原则**：模块化、可配置性、可扩展性、生产就绪的设计理念

### 2. [LlamaIndex 编排说明](./llamaindex-orchestration.md)
- **框架概述**：LlamaIndex 在 MediMind Agent 中的应用
- **核心组件集成**：文档处理、向量存储、检索生成的集成方式
- **多模态嵌入模型编排**：BioBERT 和智谱AI 嵌入的协调使用
- **混合检索编排**：多索引管理和结果融合策略
- **智能体编排**：基于 LlamaIndex 的智能体架构设计
- **配置管理编排**：YAML 配置和动态配置加载
- **持久化编排**：索引的构建、保存和加载
- **后处理编排**：自定义后处理器的实现和应用

### 3. [RAG 工作流程文档](./rag-workflow.md)
- **RAG 流程概述**：检索增强生成的完整工作流程
- **文档加载阶段**：支持 TXT、PDF、Excel、CSV 等多种格式
- **文本切片阶段**：智能文本分割和语义完整性保持
- **向量化阶段**：BioBERT（英文）和智谱AI（中文）多模态嵌入
- **索引和存储阶段**：FAISS 索引构建、多索引管理、持久化
- **检索融合阶段**：混合检索器、检索策略、结果融合
- **生成阶段**：智谱AI GLM-4 集成和上下文构建
- **后处理阶段**：证据一致性检查、源过滤、去重处理
- **性能优化策略**：批处理、缓存、异步处理
- **错误处理和监控**：异常处理、性能监控、日志记录

### 4. [API 使用文档](./api-usage.md)
- **快速开始**：环境配置、依赖安装、基础使用示例
- **核心 API 接口**：智能体接口、RAG 组件接口、后处理器接口
- **高级使用**：自定义嵌入模型、检索策略、后处理器
- **配置管理**：环境变量、YAML 配置、运行时配置
- **错误处理**：异常类型、错误处理示例、安全查询
- **性能优化**：批处理、缓存、异步处理
- **监控和日志**：日志配置、性能监控、部署指南

## 项目重新组织方案

### 建议的目录结构
```
MediMind-Agent/
├── src/
│   ├── rag/                    # RAG 核心模块
│   │   ├── document_loader/     # 文档加载 (txt, pdf, excel, csv)
│   │   ├── text_splitter/       # 文本切片
│   │   ├── embeddings/          # 向量化 (BioBERT + 智谱AI)
│   │   ├── indexing/            # 索引和存储 (FAISS)
│   │   ├── retrieval/           # 检索融合 (FusionRetriever)
│   │   ├── generation/          # 生成 (智谱GLM)
│   │   └── postprocessors/      # 后处理
│   ├── agent/                  # 智能体模块
│   └── common/                  # 公共组件
├── configs/                     # 配置文件
├── data/                        # 数据文件
└── docs/                        # 文档
```

### 重新组织优势
1. **逻辑一致性**：嵌入模型归属于 RAG 流程，职责更清晰
2. **模块化设计**：每个 RAG 步骤独立成包，便于维护和扩展
3. **清晰的责任划分**：每个包负责 RAG 流程中的一个特定步骤
4. **便于测试**：独立的模块便于单元测试和集成测试

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置 API 密钥等
```

### 2. 基础使用
```python
from src.llm.zhipu import build_llm
from src.agent.agent import MediMindAgent

# 初始化智能体
llm = build_llm(model="glm-4", temperature=0.2, max_tokens=2048)
agent = MediMindAgent(llm=llm)

# 进行问答
response = agent.query("什么是糖尿病？")
print(response)
```

### 3. 带 RAG 的问答
```python
from src.rag.indexing import load_index
from src.rag.query_engine import build_query_engine
from src.embeddings.zhipu import build_embedding

# 加载索引和查询引擎
embed_model = build_embedding()
index = load_index("data/indexes/zhipu", embed_model)
query_engine = build_query_engine(index, llm, top_k=5)

# 创建带 RAG 的智能体
agent = MediMindAgent(llm=llm, query_engine=query_engine)
response = agent.query("糖尿病的治疗方法有哪些？")
```

## 技术特色

### 1. 多模态嵌入
- **BioBERT**：专门针对英文生物医学文本优化
- **智谱AI Embedding-3**：针对中文医学文档优化
- **混合检索**：根据文档语言自动选择最佳嵌入模型

### 2. 智能检索融合
- **多索引管理**：同时维护中英文医学文档索引
- **结果融合**：智能融合不同嵌入模型的检索结果
- **权重调节**：可配置的检索结果权重

### 3. 生产就绪设计
- **LlamaIndex 框架**：成熟的 RAG 框架，确保稳定性
- **配置管理**：YAML 配置文件，支持环境变量覆盖
- **错误处理**：完善的异常处理和降级策略
- **性能优化**：批处理、缓存、异步处理支持

### 4. 可扩展架构
- **模块化设计**：每个组件独立，便于替换和扩展
- **插件化支持**：支持自定义嵌入模型、检索策略、后处理器
- **配置驱动**：通过配置文件控制系统行为

## 开发指南

### 1. 添加新的嵌入模型
```python
# 继承 BaseEmbedding 接口
class CustomEmbedding(BaseEmbedding):
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 实现自定义嵌入逻辑
        pass
```

### 2. 添加新的检索策略
```python
# 实现检索策略接口
class CustomRetrievalStrategy:
    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        # 实现自定义检索逻辑
        pass
```

### 3. 添加新的后处理器
```python
# 继承 BaseNodePostprocessor 接口
class CustomPostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None) -> List[NodeWithScore]:
        # 实现自定义后处理逻辑
        return nodes
```

## 贡献指南

1. **代码规范**：遵循 PEP 8 代码规范
2. **文档更新**：新增功能需要更新相应文档
3. **测试覆盖**：新功能需要添加相应的测试用例
4. **配置管理**：新功能需要支持配置文件管理

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 参与讨论

---

通过阅读这些文档，您将能够全面了解 MediMind Agent 的设计理念、技术架构、使用方法和扩展方式。我们建议按照文档的顺序阅读，从架构文档开始，逐步深入了解系统的各个方面。