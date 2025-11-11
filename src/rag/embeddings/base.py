"""
Embedding 模型基础类

定义统一的 embedding 接口，遵循开放封闭原则 (OCP)
"""

from abc import ABC, abstractmethod
from typing import List
from llama_index.core.embeddings import BaseEmbedding


class BaseEmbeddingModel(BaseEmbedding):
    """Embedding 模型抽象基类

    所有自定义 embedding 模型应继承此类，确保接口一致性
    遵循单一职责原则 (SRP)：仅定义 embedding 的通用接口
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """返回嵌入向量的维度

        Returns:
            int: 向量维度（如 768, 1024 等）
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称，用于日志和调试

        Returns:
            str: 模型名称（如 'biobert-v1.1', 'zhipu-embedding-3'）
        """
        pass

    @abstractmethod
    def _get_query_embedding(self, query: str) -> List[float]:
        """为查询文本生成嵌入向量

        Args:
            query: 查询文本

        Returns:
            嵌入向量
        """
        pass

    @abstractmethod
    def _get_text_embedding(self, text: str) -> List[float]:
        """为单个文本生成嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        pass

    @abstractmethod
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        pass

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询嵌入（默认使用同步实现）

        子类可覆盖此方法以提供真正的异步实现
        """
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本嵌入（默认使用同步实现）

        子类可覆盖此方法以提供真正的异步实现
        """
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文本嵌入（默认使用同步实现）

        子类可覆盖此方法以提供真正的异步实现
        """
        return self._get_text_embeddings(texts)
