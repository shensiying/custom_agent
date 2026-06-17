# retrievers.py — 多路检索：稠密向量 + BM25 稀疏检索 + 混合融合
import pickle
import jieba
from pathlib import Path
from typing import List, Tuple
from collections import OrderedDict

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import CHROMA_DIR


# ============================================================
# BM25 稀疏检索器
# ============================================================
class BM25Retriever:
    """基于 BM25 的稀疏检索器，纯 Python 实现。"""

    def __init__(self, docs: List[Document] = None):
        self._docs: List[Document] = []
        self._contents: List[str] = []
        self._tokens: List[List[str]] = []
        self._bm25: BM25Okapi = None
        if docs:
            self.index(docs)

    def index(self, docs: List[Document]):
        """用 jieba 分词构建 BM25 索引。"""
        self._docs = list(docs)
        self._contents = [d.page_content for d in docs]
        self._tokens = [list(jieba.cut(c)) for c in self._contents]
        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """BM25 检索，返回 (Document, score) 列表。"""
        if self._bm25 is None:
            return []
        query_tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(query_tokens)
        # 取 top_k
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in indices:
            if scores[idx] > 0:
                results.append((self._docs[idx], float(scores[idx])))
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"docs": self._docs, "contents": self._contents, "tokens": self._tokens}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._docs = data["docs"]
        obj._contents = data["contents"]
        obj._tokens = data["tokens"]
        obj._bm25 = BM25Okapi(obj._tokens)
        return obj


# ============================================================
# 混合检索器：稠密 + 稀疏 融合
# ============================================================
class HybridRetriever:
    """
    混合检索器：
    1. 稠密检索（ChromaDB similarity_search）
    2. BM25 稀疏检索
    3. RRF (Reciprocal Rank Fusion) 融合排序
    """

    def __init__(self, vectorstore, bm25: BM25Retriever,
                 dense_weight: float = 0.6, sparse_weight: float = 0.4,
                 dense_k: int = 10, sparse_k: int = 10):
        self.vs = vectorstore
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.dense_k = dense_k
        self.sparse_k = sparse_k

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        混合检索，返回融合排序后的 Document 列表。
        使用 RRF (Reciprocal Rank Fusion) 进行分数融合。
        """
        # 1. 稠密检索
        dense_results = self.vs.similarity_search(query, k=self.dense_k)
        dense_ranks = {self._doc_key(d): i + 1 for i, d in enumerate(dense_results)}

        # 2. BM25 稀疏检索
        sparse_results = self.bm25.search(query, top_k=self.sparse_k)
        sparse_ranks = {self._doc_key(d): i + 1 for i, (d, _) in enumerate(sparse_results)}

        # 3. RRF 融合
        all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        k = 60  # RRF 常数
        scores = {}
        doc_map = {}

        for d in dense_results:
            key = self._doc_key(d)
            doc_map[key] = d
        for d, _ in sparse_results:
            key = self._doc_key(d)
            doc_map[key] = d

        for key in all_keys:
            score = 0.0
            if key in dense_ranks:
                score += self.dense_weight / (k + dense_ranks[key])
            if key in sparse_ranks:
                score += self.sparse_weight / (k + sparse_ranks[key])
            scores[key] = score

        # 排序
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:top_k]]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """用内容和元数据源文件生成唯一 key。"""
        content_prefix = doc.page_content[:80] if doc.page_content else ""
        source = doc.metadata.get("source_file", "")
        return f"{source}|{content_prefix}"


# ============================================================
# BM25 索引管理
# ============================================================
_BM25_INDEX_PATH = str(Path(CHROMA_DIR) / "bm25_index.pkl")


def build_bm25_from_chroma(vectorstore) -> BM25Retriever:
    """从 ChromaDB 全量拉取文档，构建 BM25 索引。"""
    collection = vectorstore._collection
    all_data = collection.get(include=["documents", "metadatas"])
    docs = []
    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        docs.append(Document(page_content=content, metadata=meta or {}))
    bm25 = BM25Retriever(docs)
    bm25.save(_BM25_INDEX_PATH)
    return bm25


def load_or_build_bm25(vectorstore) -> BM25Retriever:
    """加载已有 BM25 索引，不存在则从 ChromaDB 构建。"""
    if Path(_BM25_INDEX_PATH).exists():
        return BM25Retriever.load(_BM25_INDEX_PATH)
    return build_bm25_from_chroma(vectorstore)
