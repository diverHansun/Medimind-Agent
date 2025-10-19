# MediMind Agent 项目架构文档

## 开发目的

MediMind Agent 是一个基于 LlamaIndex 框架构建的 RAG（检索增强生成）快速原型系统。本项目通过对 LlamaIndex 核心组件的轻量封装，实现文档加载、向量化、索引存储、检索和生成的完整流程。

### 核心目标

1. **快速原型**：基于 LlamaIndex 快速构建可用的 RAG 系统
2. **模块化封装**：对 LlamaIndex 组件进行简洁封装，便于配置和扩展
3. **清晰架构**：明确的数据流转路径和模块职责划分
4. **生产就绪**：完善的配置管理、错误处理和监控机制

## 架构选择

### 1. 框架选择：LlamaIndex

**选择理由**：
- **成熟稳定**：LlamaIndex 是业界领先的 RAG 框架，具有完善的生态系统
- **模块化设计**：提供清晰的组件分离，便于定制和扩展
- **生产就绪**：内置持久化、缓存、监控等生产环境必需功能
- **社区支持**：活跃的社区和丰富的文档资源

**核心优势**：
- 统一的文档处理接口
- 内置的向量存储抽象
- 灵活的检索和生成组件
- 完善的配置管理系统

### 2. 嵌入模型选择：智谱AI Embedding-3

**选择理由**：
- **API 集成**：通过 API 调用，无需本地部署
- **高维度**：1024 维向量，提供丰富的语义表示
- **官方支持**：LlamaIndex 提供官方集成 `ZhipuAIEmbedding`
- **灵活配置**：支持自定义维度和参数

**LlamaIndex 提供**：
- `ZhipuAIEmbedding` 类，封装了智谱AI的嵌入API
- 异步支持，批量处理能力

**我们的封装**：
- 工厂函数简化实例创建
- 配置文件管理 API Key 和参数
- 错误处理和重试机制

### 3. 向量存储选择：FAISS

**选择理由**：
- **高性能**：高效的向量相似性搜索
- **本地存储**：无需外部数据库依赖
- **LlamaIndex 集成**：`FaissVectorStore` 原生支持
- **持久化**：支持索引的保存和加载

**LlamaIndex 提供**：
- `FaissVectorStore` 类，封装 FAISS 索引操作
- `StorageContext` 管理存储上下文
- 自动持久化和加载机制

**我们的封装**：
- 索引构建和管理的统一接口
- 配置化的索引参数
- 索引版本管理

### 4. 大语言模型选择：智谱AI GLM-4

**选择理由**：
- **API 集成**：通过 API 调用，无需本地部署
- **中文优化**：针对中文场景优化
- **官方支持**：LlamaIndex 提供官方集成
- **稳定可靠**：提供稳定的 API 服务

**LlamaIndex 提供**：
- `ZhipuAI` LLM 类，封装智谱AI的生成API
- 流式输出支持
- 异步调用能力

**我们的封装**：
- 工厂函数简化LLM实例创建
- 配置化的模型参数（temperature、max_tokens等）
- 统一的错误处理

## 系统架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    MediMind Agent                           │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer（智能体层）                                     │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  MediMindAgent  │    │   Controller    │                │
│  │  (协调RAG流程)   │    │  (对话管理)      │                │
│  └─────────────────┘    └─────────────────┘                │
├─────────────────────────────────────────────────────────────┤
│  RAG Pipeline（RAG流水线 - 对LlamaIndex的封装）              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Document    │ │ Text        │ │ Embeddings  │          │
│  │ Loader      │→│ Splitter    │→│ (ZhipuAI)   │          │
│  │ (加载)      │ │ (切片)      │ │ (向量化)     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│         ↓                                  ↓                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Indexing    │←│ Retrieval   │←│ Generation  │          │
│  │ (FAISS索引) │ │ (检索)      │ │ (GLM-4生成) │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│         ↓                ↑                                  │
│  ┌─────────────┐        │                                  │
│  │Postprocessors│───────┘                                  │
│  │ (后处理)     │                                           │
│  └─────────────┘                                           │
├─────────────────────────────────────────────────────────────┤
│  LlamaIndex Core（底层框架）                                 │
│  SimpleDirectoryReader | SentenceSplitter | ZhipuAIEmbedding│
│  FaissVectorStore | VectorStoreIndex | QueryEngine         │
└─────────────────────────────────────────────────────────────┘
```

### 数据流转路径

```
原始文件(TXT/PDF/CSV)
    ↓
[Document Loader] → Document对象列表
    ↓
[Text Splitter] → Node对象列表（文本块）
    ↓
[Embeddings] → 向量化的Node（带embedding）
    ↓
[Indexing] → FAISS向量索引
    ↓
[Retrieval] ← 用户查询 → 检索相关Node
    ↓
[Generation] → 基于检索结果生成回答
    ↓
[Postprocessors] → 优化后的最终回答
```

### 模块职责与封装说明

#### 1. document_loader/ - 文档加载模块
**LlamaIndex 提供**：`SimpleDirectoryReader`
**我们的封装**：
- 工厂函数统一加载接口
- 支持 TXT、PDF、CSV 格式
- 自动元数据提取和管理

#### 2. text_splitter/ - 文本切片模块
**LlamaIndex 提供**：`SentenceSplitter`
**我们的封装**：
- 配置化的切片参数（chunk_size、chunk_overlap）
- 针对不同文档类型的切片策略

#### 3. embeddings/ - 向量化模块
**LlamaIndex 提供**：`ZhipuAIEmbedding`
**我们的封装**：
- 工厂函数创建嵌入模型实例
- API Key 配置管理
- 批量向量化处理

#### 4. indexing/ - 索引构建模块
**LlamaIndex 提供**：`FaissVectorStore`、`VectorStoreIndex`
**我们的封装**：
- 索引构建的统一接口
- 索引持久化和加载
- 索引版本管理

#### 5. retrieval/ - 检索模块
**LlamaIndex 提供**：`VectorStoreIndex.as_retriever()`
**我们的封装**：
- 检索参数配置（top_k、similarity_cutoff）
- 检索结果格式化
- 检索性能监控

#### 6. generation/ - 生成模块
**LlamaIndex 提供**：`QueryEngine`
**我们的封装**：
- LLM 实例管理
- 提示词模板管理
- 生成参数配置

#### 7. postprocessors/ - 后处理模块
**LlamaIndex 提供**：`BaseNodePostprocessor`
**我们的封装**：
- 自定义后处理器（去重、过滤等）
- 后处理器链管理

## 技术栈

### 核心依赖
- **LlamaIndex**：RAG 框架核心
- **llama-index-embeddings-zhipuai**：智谱AI嵌入集成
- **llama-index-llms-zhipuai**：智谱AI LLM集成
- **llama-index-vector-stores-faiss**：FAISS向量存储集成
- **FAISS**：向量相似性搜索引擎

### 配置管理
- **YAML**：配置文件格式
- **Python-dotenv**：环境变量管理

### 工具库
- **PyYAML**：YAML 文件解析
- **logging**：日志记录

## 设计原则

### 1. 轻量封装
- 每个模块是对 LlamaIndex 组件的简洁封装
- 保持与 LlamaIndex 的接口一致性
- 避免过度抽象

### 2. 配置驱动
- YAML 配置文件管理所有参数
- 环境变量管理敏感信息（API Key）
- 运行时参数可调整

### 3. 清晰的数据流
- 明确的数据对象转换路径
- 每个模块的输入输出清晰定义
- 便于调试和问题定位

### 4. 生产就绪
- 完善的错误处理和降级策略
- 日志记录和性能监控
- 索引持久化和版本管理

## 部署架构

### 开发环境
- 本地文件系统存储
- 单机部署
- 开发调试工具

### 生产环境
- 分布式存储支持
- 容器化部署
- 监控和日志系统

## 性能考虑

### 1. 检索性能
- FAISS 索引优化
- 批量向量化处理
- 缓存机制

### 2. 生成性能
- 模型量化
- 推理优化
- 并发处理

### 3. 存储优化
- 索引压缩
- 增量更新
- 数据清理策略

## 安全考虑

### 1. 数据安全
- 敏感信息脱敏
- 访问权限控制
- 数据加密存储

### 2. 模型安全
- 输入验证
- 输出过滤
- 安全护栏机制

## 未来扩展

### 1. 功能扩展
- 多模态输入支持
- 实时学习能力
- 个性化推荐

### 2. 性能优化
- 模型蒸馏
- 硬件加速
- 分布式计算

### 3. 集成能力
- API 服务化
- 微服务架构
- 云原生部署
