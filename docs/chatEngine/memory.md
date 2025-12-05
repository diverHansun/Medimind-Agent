# Memory 模块设计

## 概述

Memory 模块负责管理对话内记忆，使用 LlamaIndex 的 ChatSummaryMemoryBuffer 实现。当对话历史超出 token 限制时，自动调用 LLM 对旧消息进行摘要。

## 核心组件

### ChatSummaryMemoryBuffer

LlamaIndex 提供的记忆缓冲区，具备以下特性：

| 特性 | 说明 |
|------|------|
| Token 限制 | 设定最大 token 数，超出时触发摘要 |
| 自动摘要 | 调用 LLM 对旧消息生成摘要 |
| 摘要保留 | 摘要作为系统消息保留在历史中 |
| 异步支持 | 支持同步和异步操作 |

### 工作流程

```
对话消息累积
      |
      v
检查是否超出 token_limit
      |
      +-- 未超出 --> 直接存储，继续对话
      |
      +-- 超出 --> 调用 LLM 生成摘要
                        |
                        v
                  摘要 + 最近消息 = 新的对话历史
                        |
                        v
                    继续对话
```

## 接口设计

### 模块位置

```
src/memory/
├── __init__.py
└── chat_memory.py
```

### 主要函数

#### build_chat_memory

```python
def build_chat_memory(
    llm: LLM,
    token_limit: int = 64000,
    summarize_prompt: Optional[str] = None,
    chat_store: Optional[BaseChatStore] = None,
    chat_store_key: str = "default",
) -> ChatSummaryMemoryBuffer:
    """构建对话记忆缓冲区。

    Args:
        llm: 用于生成摘要的 LLM 实例
        token_limit: 最大 token 限制
        summarize_prompt: 可选的自定义摘要提示词
        chat_store: 可选的持久化存储后端
        chat_store_key: 会话标识键

    Returns:
        ChatSummaryMemoryBuffer 实例
    """
```

#### load_memory_config

```python
def load_memory_config(config_path: str = "configs/memory.yaml") -> dict:
    """加载 Memory 配置。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
```

## 配置文件

### 路径

`configs/memory.yaml`

### 配置项

```yaml
memory:
  # Token 限制
  token_limit: 64000

  # 摘要提示词（可选，使用默认值时可省略）
  summarize_prompt: |
    请总结以下对话的关键信息，保留重要的上下文和结论。

  # 持久化配置（预留，当前不启用）
  # persistence:
  #   enabled: false
  #   backend: redis
  #   connection_string: redis://localhost:6379
```

## 使用示例

### 基础使用

```python
from src.memory import build_chat_memory
from src.llm.zhipu import build_llm

# 构建 LLM
llm = build_llm()

# 构建记忆
memory = build_chat_memory(llm=llm, token_limit=64000)

# 与 ChatEngine 集成
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm,
)
```

### 手动操作记忆

```python
from llama_index.core.llms import ChatMessage

# 获取当前历史（受 token 限制）
history = memory.get()

# 获取所有历史（不受限制）
all_history = memory.get_all()

# 添加消息
memory.put(ChatMessage(role="user", content="问题"))
memory.put(ChatMessage(role="assistant", content="回答"))

# 重置记忆
memory.reset()
```

## 与 ChatEngine 的集成

### 集成方式

ChatEngine 内部自动调用 Memory 的方法：

| 时机 | 调用方法 | 说明 |
|------|---------|------|
| 对话前 | `memory.get()` | 获取历史上下文 |
| 对话后 | `memory.put()` | 保存新消息 |
| 重置时 | `memory.reset()` | 清空历史 |

### 代码示例

```python
# ChatEngine 会自动管理 Memory
response = chat_engine.chat("什么是糖尿病？")
# Memory 自动记录这一轮对话

response = chat_engine.chat("它有哪些症状？")
# Memory 自动带入上一轮的上下文
```

## 注意事项

### Token 计算

- token_limit 包含所有对话历史的 token 总和
- 摘要生成后，摘要文本也计入 token 计算
- 建议预留一定余量，避免频繁触发摘要

### 摘要质量

- 摘要质量取决于 LLM 能力
- 可通过 summarize_prompt 自定义摘要策略
- 过于频繁的摘要可能丢失细节信息

### 会话隔离

- 当前实现为单会话模式
- 多会话场景需要为每个会话创建独立的 Memory 实例
- 后续持久化时通过 chat_store_key 区分不同会话
