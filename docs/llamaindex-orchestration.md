# LlamaIndex 编排说明文档

## LlamaIndex 框架概述

LlamaIndex 是一个专为构建 RAG（检索增强生成）应用而设计的 Python 框架。它提供了完整的工具链，从文档加载、文本处理、向量化、索引构建到检索和生成，为开发者提供了统一的接口和丰富的功能。

## 在 MediMind Agent 中的应用

### 1. 核心组件集成

#### 1.1 文档处理管道
```python
# 文档加载器
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader, ExcelReader

# 文本分割器
from llama_index.core.text_splitter import SentenceSplitter

# 嵌入模型
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
```

#### 1.2 向量存储集成
```python
# FAISS 向量存储
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# 索引构建
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    embed_model=embed_model
)
```

#### 1.3 检索和生成
```python
# 查询引擎
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    node_postprocessors=postprocessors
)
```

### 2. 多模态嵌入模型编排

#### 2.1 BioBERT 集成
```python
class BioBERTEmbedding(BaseEmbedding):
    """BioBERT 嵌入模型，继承 LlamaIndex BaseEmbedding 接口"""
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 实现 BioBERT 向量化逻辑
        pass
```

#### 2.2 智谱AI 嵌入集成
```python
# 使用 LlamaIndex 官方智谱AI 集成
zhipu_embedding = ZhipuAIEmbedding(
    model="embedding-3",
    api_key=api_key,
    dimensions=1024
)
```

### 3. 混合检索编排

#### 3.1 多索引管理
```python
class HybridRetriever:
    """混合检索器，管理多个向量索引"""
    
    def __init__(self, biobert_index, zhipu_index):
        self.biobert_index = biobert_index
        self.zhipu_index = zhipu_index
    
    def retrieve(self, query: str, top_k: int = 5):
        # 从多个索引检索并融合结果
        biobert_nodes = self.biobert_index.as_retriever().retrieve(query)
        zhipu_nodes = self.zhipu_index.as_retriever().retrieve(query)
        
        # 结果融合和排序
        return self._merge_results(biobert_nodes, zhipu_nodes)
```

#### 3.2 结果融合策略
```python
def _merge_results(self, biobert_nodes, zhipu_nodes):
    """融合多个检索结果"""
    # 权重加权融合
    for node in biobert_nodes:
        node.score *= self.weights['biobert']
    
    for node in zhipu_nodes:
        node.score *= self.weights['zhipu']
    
    # 合并并排序
    all_nodes = biobert_nodes + zhipu_nodes
    all_nodes.sort(key=lambda x: x.score, reverse=True)
    
    return all_nodes[:top_k]
```

### 4. 智能体编排

#### 4.1 智能体架构
```python
class MediMindAgent:
    """基于 LlamaIndex 的医学智能体"""
    
    def __init__(self, llm, query_engine=None, safety=None):
        self.llm = llm
        self.query_engine = query_engine  # LlamaIndex QueryEngine
        self.safety = safety
    
    def query(self, question: str):
        """处理用户查询"""
        if self.query_engine:
            # 使用 RAG 流程
            response = self.query_engine.query(question)
            return str(response)
        else:
            # 直接 LLM 生成
            response = self.llm.complete(question)
            return str(response)
```

#### 4.2 对话管理
```python
def chat(self, messages: List[ChatMessage]):
    """处理对话上下文"""
    # 更新对话历史
    self.conversation_history.extend(messages)
    
    # 使用 LlamaIndex 的聊天引擎
    if self.query_engine:
        response = self.query_engine.query(messages[-1].content)
    else:
        response = self.llm.chat(messages)
    
    return response
```

### 5. 配置管理编排

#### 5.1 YAML 配置集成
```yaml
# configs/embeddings.yaml
embeddings:
  biobert:
    enabled: true
    model: biobert-sentence
    dim: 768
  zhipu:
    enabled: true
    model: embedding-3
    dim: 1024

merge:
  strategy: score_weighted
  weights:
    biobert: 1.0
    zhipu: 1.0
```

#### 5.2 动态配置加载
```python
def get_embeddings_config():
    """从 YAML 文件加载嵌入配置"""
    config_path = "configs/embeddings.yaml"
    return load_yaml_config(config_path)

def build_hybrid_retriever(documents, config):
    """根据配置构建混合检索器"""
    if config['embeddings']['biobert']['enabled']:
        biobert_embed = build_biobert_embed()
        biobert_index = build_index(documents, biobert_embed)
    
    if config['embeddings']['zhipu']['enabled']:
        zhipu_embed = build_zhipu_embed()
        zhipu_index = build_index(documents, zhipu_embed)
    
    return HybridRetriever(biobert_index, zhipu_index)
```

### 6. 持久化编排

#### 6.1 索引持久化
```python
def build_index(documents, embed_model, persist_dir=None):
    """构建并持久化索引"""
    # 创建 FAISS 索引
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 构建索引
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # 持久化
    if persist_dir:
        index.storage_context.persist(persist_dir=persist_dir)
    
    return index
```

#### 6.2 索引加载
```python
def load_index(persist_dir, embed_model):
    """加载持久化的索引"""
    vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=persist_dir
    )
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    return index
```

### 7. 后处理编排

#### 7.1 自定义后处理器
```python
class EvidenceConsistencyChecker(BaseNodePostprocessor):
    """证据一致性检查器"""
    
    def _postprocess_nodes(self, nodes, query_bundle=None):
        # 实现证据一致性检查逻辑
        return filtered_nodes

class SourceFilterPostprocessor(BaseNodePostprocessor):
    """源过滤后处理器"""
    
    def _postprocess_nodes(self, nodes, query_bundle=None):
        # 根据源类型过滤节点
        return filtered_nodes
```

#### 7.2 后处理器链
```python
def build_query_engine(index, llm, postprocessors=None):
    """构建带后处理器的查询引擎"""
    if postprocessors is None:
        postprocessors = []
    
    # 添加自定义后处理器
    postprocessors.extend([
        EvidenceConsistencyChecker(),
        SourceFilterPostprocessor(preferred_sources=['guideline', 'pubmed'])
    ])
    
    return index.as_query_engine(
        llm=llm,
        node_postprocessors=postprocessors
    )
```

### 8. 错误处理和监控

#### 8.1 异常处理
```python
def safe_query(agent, question):
    """安全的查询处理"""
    try:
        response = agent.query(question)
        return response
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return "抱歉，处理您的问题时出现了错误。"
```

#### 8.2 性能监控
```python
import time
from functools import wraps

def monitor_performance(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper
```

### 9. 扩展性设计

#### 9.1 插件化架构
```python
class EmbeddingPlugin:
    """嵌入模型插件接口"""
    
    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError
    
    def get_dimension(self) -> int:
        raise NotImplementedError

class CustomEmbedding(EmbeddingPlugin):
    """自定义嵌入模型"""
    
    def get_embedding(self, text: str) -> List[float]:
        # 实现自定义嵌入逻辑
        pass
```

#### 9.2 策略模式
```python
class RetrievalStrategy:
    """检索策略接口"""
    
    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        raise NotImplementedError

class HybridRetrievalStrategy(RetrievalStrategy):
    """混合检索策略"""
    
    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        # 实现混合检索逻辑
        pass
```

## LlamaIndex 编排优势

### 1. 统一接口
- 所有组件都遵循 LlamaIndex 标准接口
- 便于组件替换和扩展
- 减少学习成本

### 2. 内置优化
- 自动的批处理优化
- 内存管理优化
- 并发处理支持

### 3. 生产就绪
- 完善的错误处理机制
- 内置的日志记录
- 性能监控支持

### 4. 社区支持
- 丰富的文档和示例
- 活跃的社区支持
- 定期的版本更新

## 最佳实践

### 1. 组件设计
- 保持组件的单一职责
- 使用依赖注入模式
- 实现清晰的接口定义

### 2. 配置管理
- 使用 YAML 配置文件
- 支持环境变量覆盖
- 实现配置验证

### 3. 错误处理
- 实现优雅的降级策略
- 提供详细的错误信息
- 记录完整的执行日志

### 4. 性能优化
- 使用批量处理
- 实现缓存机制
- 监控关键性能指标

通过 LlamaIndex 框架的编排，MediMind Agent 实现了高度模块化、可扩展和可维护的 RAG 系统架构，为医学问答应用提供了强大的技术基础。
