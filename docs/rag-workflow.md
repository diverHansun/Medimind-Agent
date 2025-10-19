# RAG 工作流程文档

## RAG 流程概述

本文档详细说明 MediMind Agent 的 RAG（检索增强生成）工作流程，重点阐述数据在各模块间的流转方式、LlamaIndex 提供的能力以及我们的封装策略。

### 核心流程

RAG 流程将文档处理分为七个阶段，每个阶段对应一个独立模块，数据对象在模块间依次转换和传递。

## 完整 RAG 工作流程

### 1. 文档加载阶段

**数据流转**：
```
原始文件(TXT/PDF/CSV) → Document对象列表
```

**LlamaIndex 提供**：
- `SimpleDirectoryReader`：支持多种文件格式的自动加载
- 自动文件类型检测（基于扩展名）
- 自动元数据提取（file_path、file_name、file_type、file_size、creation_date等）
- 并行加载支持
- 递归目录读取

**我们的封装**（`src/rag/document_loader/`）：
- 工厂函数简化加载接口
- 配置化的文件格式过滤（required_exts）
- 自定义元数据增强（添加业务相关元数据）
- 统一的错误处理

**关键配置参数**：
- `input_dir`：文档目录路径
- `recursive`：是否递归读取子目录
- `required_exts`：限制文件类型（[".txt", ".pdf", ".csv"]）
- `file_metadata`：自定义元数据提取函数

### 2. 文本切片阶段

**数据流转**：
```
Document对象列表 → Node对象列表（文本块）
```

**LlamaIndex 提供**：
- `SentenceSplitter`：按句子边界智能分割
- `TokenTextSplitter`：按token数量分割
- 自动保持语义完整性（不在句子中间切分）
- chunk_overlap 支持（保持上下文连贯性）
- 自动继承 Document 的元数据

**我们的封装**（`src/rag/text_splitter/`）：
- 配置化的切片参数
- 针对不同文档类型的切片策略
- 切片质量监控（平均长度、重叠率等）

**关键配置参数**：
- `chunk_size`：文本块大小（默认512，可根据需求调整）
- `chunk_overlap`：重叠大小（默认50，保持上下文）
- `separator`：分隔符（默认按句子）

**切片策略选择**：
- 小chunk_size（256-512）：更精确的检索，但可能丢失上下文
- 大chunk_size（1024-2048）：更完整的上下文，但检索精度降低
- 建议：从512开始，根据实际效果调整

### 3. 向量化阶段

**数据流转**：
```
Node对象列表 → 带embedding向量的Node列表
```

**LlamaIndex 提供**：
- `ZhipuAIEmbedding`：智谱AI官方集成
- 自动批量处理
- 异步支持（`aget_text_embeddings`）
- 自动将embedding附加到Node对象

**我们的封装**（`src/rag/embeddings/`）：
- 工厂函数创建嵌入模型实例
- API Key 配置管理（从环境变量或配置文件读取）
- 错误处理和重试机制
- 向量化性能监控

**关键配置参数**：
- `model`：模型名称（"embedding-3"）
- `api_key`：智谱AI API密钥
- `dimensions`：向量维度（1024）
- `timeout`：API超时时间

**性能优化**：
- 批量向量化（减少API调用次数）
- 异步处理（提高并发性能）
- 缓存机制（避免重复向量化）

### 4. 索引和存储阶段

#### 4.1 FAISS 索引构建
```python
# src/rag/indexing/
def build_faiss_index(documents: List[Document], embed_model: BaseEmbedding) -> VectorStoreIndex:
    """构建 FAISS 索引"""
    # 创建 FAISS 索引
    faiss_index = faiss.IndexFlatIP(embed_model.dimensions)
    
    # 创建向量存储
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    # 创建存储上下文
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 构建索引
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    return index
```

#### 4.2 多索引管理
```python
class MultiIndexManager:
    """多索引管理器"""
    
    def __init__(self):
        self.indexes = {}
    
    def add_index(self, name: str, index: VectorStoreIndex):
        """添加索引"""
        self.indexes[name] = index
    
    def get_index(self, name: str) -> VectorStoreIndex:
        """获取索引"""
        return self.indexes.get(name)
    
    def list_indexes(self) -> List[str]:
        """列出所有索引"""
        return list(self.indexes.keys())
```

#### 4.3 索引持久化
```python
def persist_index(index: VectorStoreIndex, persist_dir: str):
    """持久化索引"""
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)

def load_index(persist_dir: str, embed_model: BaseEmbedding) -> VectorStoreIndex:
    """加载索引"""
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

### 5. 检索融合阶段

#### 5.1 混合检索器
```python
# src/rag/retrieval/
class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, biobert_index: VectorStoreIndex, zhipu_index: VectorStoreIndex):
        self.biobert_index = biobert_index
        self.zhipu_index = zhipu_index
        self.weights = {"biobert": 1.0, "zhipu": 1.0}
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """混合检索"""
        # 从两个索引检索
        biobert_nodes = self.biobert_index.as_retriever().retrieve(query)
        zhipu_nodes = self.zhipu_index.as_retriever().retrieve(query)
        
        # 应用权重
        for node in biobert_nodes:
            node.score *= self.weights["biobert"]
        for node in zhipu_nodes:
            node.score *= self.weights["zhipu"]
        
        # 合并结果
        all_nodes = biobert_nodes + zhipu_nodes
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return all_nodes[:top_k]
```

#### 5.2 检索策略
```python
class RetrievalStrategy:
    """检索策略接口"""
    
    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        raise NotImplementedError

class ScoreWeightedStrategy(RetrievalStrategy):
    """分数加权策略"""
    
    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        # 实现分数加权检索
        pass

class InterleaveStrategy(RetrievalStrategy):
    """交替策略"""
    
    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        # 实现交替检索
        pass
```

### 6. 生成阶段

#### 6.1 智谱AI GLM-4 集成
```python
# src/rag/generation/
class ZhipuAIGenerator:
    """智谱AI 生成器"""
    
    def __init__(self, model: str = "glm-4", api_key: str = None):
        self.model = model
        self.api_key = api_key
        self.client = ZhipuAI(api_key=api_key)
    
    def generate(self, query: str, context: str) -> str:
        """生成回答"""
        prompt = self._build_prompt(query, context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048
        )
        
        return response.choices[0].message.content
```

#### 6.2 上下文构建
```python
def build_context(query: str, retrieved_nodes: List[NodeWithScore]) -> str:
    """构建生成上下文"""
    context_parts = []
    
    for i, node in enumerate(retrieved_nodes, 1):
        source = node.node.metadata.get("source", "unknown")
        text = node.node.text
        score = node.score
        
        context_parts.append(f"[来源 {i}: {source}, 相关性: {score:.3f}]\n{text}")
    
    return "\n\n".join(context_parts)
```

### 7. 后处理阶段

#### 7.1 证据一致性检查
```python
# src/rag/postprocessors/
class EvidenceConsistencyChecker(BaseNodePostprocessor):
    """证据一致性检查器"""
    
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None) -> List[NodeWithScore]:
        """检查证据一致性"""
        if len(nodes) < 2:
            return nodes
        
        # 计算节点间的一致性分数
        consistency_scores = self._calculate_consistency(nodes)
        
        # 过滤低一致性节点
        filtered_nodes = [
            node for node, score in zip(nodes, consistency_scores)
            if score >= self.consistency_threshold
        ]
        
        return filtered_nodes
```

#### 7.2 源过滤
```python
class SourceFilterPostprocessor(BaseNodePostprocessor):
    """源过滤后处理器"""
    
    def __init__(self, preferred_sources: List[str], blocked_sources: List[str] = None):
        self.preferred_sources = preferred_sources
        self.blocked_sources = blocked_sources or []
    
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None) -> List[NodeWithScore]:
        """过滤节点"""
        filtered_nodes = []
        
        for node in nodes:
            source = node.node.metadata.get("source", "").lower()
            
            # 跳过被屏蔽的源
            if source in self.blocked_sources:
                continue
            
            # 优先选择偏好的源
            if source in self.preferred_sources:
                node.score *= 1.2  # 提升分数
            
            filtered_nodes.append(node)
        
        return filtered_nodes
```

#### 7.3 去重处理
```python
class DeduplicationPostprocessor(BaseNodePostprocessor):
    """去重后处理器"""
    
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle=None) -> List[NodeWithScore]:
        """去除重复节点"""
        seen_texts = set()
        unique_nodes = []
        
        for node in nodes:
            text_hash = hash(node.node.text)
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_nodes.append(node)
        
        return unique_nodes
```

## RAG 工作流程配置

### 1. 配置文件结构
```yaml
# configs/rag.yaml
rag:
  mode: auto
  top_k:
    default: 5
    high_risk: 10
    low_risk: 3
  min_relevance: 0.25
  rerank:
    enabled: false
  time_decay:
    enabled: true
    half_life_days: 365
  filters:
    prefer_sources: [guideline, pubmed]
    blocklist: []
  require_citations_for_medical: true
```

### 2. 动态配置加载
```python
def load_rag_config() -> dict:
    """加载 RAG 配置"""
    config_path = "configs/rag.yaml"
    return load_yaml_config(config_path)

def apply_rag_config(query_engine, config: dict):
    """应用 RAG 配置"""
    # 设置检索参数
    query_engine.similarity_top_k = config.get("top_k", {}).get("default", 5)
    
    # 设置相关性阈值
    similarity_cutoff = config.get("min_relevance", 0.25)
    
    # 添加后处理器
    postprocessors = []
    if config.get("filters", {}).get("prefer_sources"):
        postprocessors.append(SourceFilterPostprocessor(
            preferred_sources=config["filters"]["prefer_sources"]
        ))
    
    return query_engine
```

## 性能优化策略

### 1. 批处理优化
```python
def batch_embedding(texts: List[str], embed_model: BaseEmbedding, batch_size: int = 32) -> List[List[float]]:
    """批量嵌入处理"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embed_model.get_text_embeddings(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

### 2. 缓存机制
```python
class EmbeddingCache:
    """嵌入缓存"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, text: str) -> Optional[List[float]]:
        """获取缓存"""
        return self.cache.get(text)
    
    def set(self, text: str, embedding: List[float]):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # 清理最旧的缓存
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text] = embedding
```

### 3. 异步处理
```python
import asyncio

async def async_retrieve(retriever, query: str, top_k: int) -> List[NodeWithScore]:
    """异步检索"""
    return await retriever.aretrieve(query, top_k=top_k)

async def async_generate(llm, prompt: str) -> str:
    """异步生成"""
    return await llm.acomplete(prompt)
```

## 错误处理和监控

### 1. 异常处理
```python
def safe_rag_query(agent, query: str) -> str:
    """安全的 RAG 查询"""
    try:
        response = agent.query(query)
        return response
    except EmbeddingError as e:
        logger.error(f"Embedding error: {str(e)}")
        return "抱歉，文档处理出现错误。"
    except RetrievalError as e:
        logger.error(f"Retrieval error: {str(e)}")
        return "抱歉，检索相关文档时出现错误。"
    except GenerationError as e:
        logger.error(f"Generation error: {str(e)}")
        return "抱歉，生成回答时出现错误。"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return "抱歉，系统出现未知错误。"
```

### 2. 性能监控
```python
import time
from functools import wraps

def monitor_rag_performance(func):
    """RAG 性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"RAG {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper
```

通过以上完整的 RAG 工作流程，MediMind Agent 实现了高效、准确、可扩展的医学问答系统，为用户提供可靠的医学知识检索和生成服务。
