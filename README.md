# MediMind-Agent 🏥

一个智能医疗健康问答助手，支持多轮对话和上下文记忆，能够基于您的医疗文档库提供准确的健康咨询。

## ✨ 功能特性

- **💬 智能对话**：支持多轮连续对话，理解上下文，像真人一样交流
- **🧠 记忆系统**：自动记住对话历史，超出限制时智能总结关键信息
- **📚 知识检索**：从您的文档库中快速找到相关信息，提供可靠回答
- **⚡ 开箱即用**：首次运行自动处理文档，之后快速启动
- **📁 支持多种文档**：TXT、PDF、CSV、Markdown、Excel、Word 等格式
- **⚙️ 灵活配置**：简单的配置文件，轻松调整参数

## 🚀 快速开始

### 1. 安装依赖

**推荐使用 uv**

```bash
pip install uv
uv sync
.venv\Scripts\activate  # Windows
```

**或使用 pip**

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

复制 `.env.example` 为 `.env`，然后填入您的 ZhipuAI API 密钥：

```bash
cp .env.example .env
# 编辑 .env 文件，填入: ZHIPU_API_KEY=your_key_here
```

### 3. 准备文档

将医疗相关文档放入 `data/documents/` 目录，按类型分类即可：

```
data/documents/
├── txt/       # 文本文件
├── pdf/       # PDF 文档
├── md/        # Markdown 文件
└── excel/     # Excel 表格
...
```

### 4. 启动应用

```bash
python main.py
```

启动后，开始和助手对话吧！输入 `quit` 或 `exit` 退出。

## 💡 使用体验

启动后，您可以像这样与助手对话：

```
User: 什么是糖尿病？
Assistant: [基于文档库回答糖尿病的定义和特点]

User: 它有哪些常见症状？
Assistant: [理解"它"指糖尿病，回答症状]

User: 如何预防？
Assistant: [继续讨论糖尿病的预防方法]
```

助手会记住对话内容，自动理解您的问题背景。

## ⚙️ 配置说明

所有配置文件在 `configs/` 目录中，您可以根据需要调整：

- **`llm.yaml`** - 语言模型参数（温度、最大长度等）
- **`embeddings.yaml`** - 文本向量化配置
- **`rag.yaml`** - 检索参数（检索数量、相似度阈值等）
- **`memory.yaml`** - 对话记忆配置（历史长度限制等）

大多数情况下使用默认配置即可，如需调整请参考配置文件中的注释。

**配置优先级**：函数参数 > 环境变量 > 配置文件 > 默认值

## 📚 进一步了解

想深入了解技术细节？查看以下文档：

- [对话引擎架构](docs/chatEngine/architecture.md) - 了解对话系统设计
- [记忆系统详解](docs/chatEngine/memory.md) - 了解对话记忆机制
- [技术开发指南](medimind-guide.md) - 开发和扩展指南

## 系统要求

- Python 3.10 或更高版本
- ZhipuAI API 密钥（[申请地址](https://open.bigmodel.cn/)）

---

