# ChatEngine 架构设计

## 概述

本文档描述 MediMind Agent 的对话引擎（ChatEngine）架构设计，用于实现带有对话内记忆的 RAG 问答功能。

## 设计目标

1. 实现对话内记忆，支持多轮上下文关联
2. 基于 LlamaIndex ChatEngine，与现有 RAG 模块无缝集成
3. 使用 ChatSummaryMemoryBuffer 管理记忆，超出限制时自动摘要
4. 为后续持久化存储预留扩展接口

## 架构总览

```
                        MediMindAgent
                             |
              +--------------+--------------+
              |                             |
        QueryEngine                   ChatEngine
        (单次查询)                  (多轮对话 + 记忆)
              |                             |
              |                    +--------+--------+
              |                    |                 |
              |              ChatMemory         Retriever
              |         (ChatSummaryMemoryBuffer)    |
              |                    |                 |
              +--------------------+-----------------+
                                   |
                            VectorStoreIndex
                                   |
                              FAISS 向量库
```

## 模块职责

### ChatEngine

- 位置：`src/rag/generation/chat_engine.py`
- 职责：
  - 封装 LlamaIndex ChatEngine 的创建逻辑
  - 集成 Memory 模块管理对话历史
  - 使用 `condense_plus_context` 模式处理多轮对话

### Memory

- 位置：`src/memory/chat_memory.py`
- 职责：
  - 封装 ChatSummaryMemoryBuffer
  - 管理对话历史的存储与检索
  - 超出 token 限制时自动调用 LLM 生成摘要

### Agent

- 位置：`src/agent/agent.py`
- 职责：
  - 统一入口，兼容 QueryEngine 和 ChatEngine
  - 协调 LLM、Engine、Safety 等组件

## 数据流

### 多轮对话流程

```
1. 用户输入问题
       |
       v
2. Memory.get() 获取对话历史
       |
       v
3. Condense: 历史 + 新问题 -> 压缩成独立问题
       |
       v
4. Retriever: 从向量库检索相关文档
       |
       v
5. LLM: 上下文 + 检索结果 + 问题 -> 生成回答
       |
       v
6. Memory.put() 保存本轮对话
       |
       v
7. 返回回答给用户
```

### chat_mode 说明

本项目使用 `condense_plus_context` 模式：

| 步骤 | 说明 |
|------|------|
| Condense | 将对话历史与新问题压缩成一个独立的查询问题 |
| Retrieve | 使用压缩后的问题进行向量检索 |
| Context | 将检索结果与对话历史一起作为上下文 |
| Generate | LLM 基于完整上下文生成回答 |

## 配置管理

### 配置文件

- 路径：`configs/memory.yaml`
- 内容：token 限制、摘要提示词等参数

### 配置优先级

```
函数参数 > 环境变量 > YAML 配置 > 默认值
```

## 扩展性设计

### 持久化预留

当前实现使用内存存储，后续可通过以下方式扩展：

```python
# 当前：内存存储
memory = ChatSummaryMemoryBuffer.from_defaults(token_limit=64000)

# 未来：持久化存储
memory = ChatSummaryMemoryBuffer.from_defaults(
    token_limit=64000,
    chat_store=RedisChatStore(...),  # 或其他存储
    chat_store_key="session_id"
)

```

### 支持的存储后端

LlamaIndex 支持以下 ChatStore：

- SimpleChatStore（内存，当前使用）
- RedisChatStore
- PostgresChatStore
- AzureChatStore
- 其他第三方实现

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/memory/__init__.py` | Memory 模块入口 |
| `src/memory/chat_memory.py` | ChatSummaryMemoryBuffer 封装 |
| `src/rag/generation/chat_engine.py` | ChatEngine 构建函数 |
| `src/agent/agent.py` | Agent 集成 |
| `configs/memory.yaml` | Memory 配置文件 |
