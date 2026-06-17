# reranker.py — Cross-Encoder 重排序
from typing import List
from langchain_core.documents import Document

import numpy as np

from config import RERANKER_MODEL

_RERANKER = None


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _RERANKER


def rerank(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    用 Cross-Encoder 对候选文档重排序。
    输入：原始候选文档列表
    输出：按相关性分数降序排列的 top_k 文档
    """
    if len(docs) <= 1:
        return docs

    model = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()

    # 按分数降序排列
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
