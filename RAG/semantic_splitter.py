# semantic_splitter.py — 基于嵌入相似度的语义切分
"""
通过句子间余弦相似度检测「主题转折点」，在语义边界切分文档。
相比固定长度的 RecursiveCharacterTextSplitter，语义切分能：
  - 自动在同属一个话题的段落内聚合
  - 在话题切换处自然断开
  - 避免把一句话拆成两半

算法：
  1. 将文档按句子分割（保留 。！？等标点作为句子边界）
  2. 对每个句子计算嵌入向量
  3. 计算相邻句子间的余弦相似度
  4. 找出相似度低于阈值的「断点」（breakpoints）
  5. 在断点处将句子组合成语义块

阈值模式：
  - percentile: 使用所有相邻相似度的 P 分位数作为阈值（适应不同文档）
  - 默认 P=50（中位数），即相似度低于中位数的相邻句被断开
"""
import re
import numpy as np
from typing import List, Optional, Literal
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# 中文句子分隔正则（保留分隔符跟在句尾）
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？\n])(?=[^。！？\n])')


def _split_sentences(text: str) -> List[str]:
    """将文本分割为句子列表（保留分隔符在句尾）。"""
    parts = _SENTENCE_SPLIT_RE.split(text)
    sentences = []
    for p in parts:
        stripped = p.strip()
        if stripped:
            sentences.append(stripped)
    return sentences


def _join_sentences(sentences: List[str]) -> str:
    """将句子列表重新拼接，句子间用空格分隔（原分隔符已在句尾保留）。"""
    return " ".join(s.strip() for s in sentences)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度。"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _compute_breakpoints(
    similarities: List[float],
    threshold_type: Literal["percentile"],
    percentile: float,
) -> List[int]:
    """
    根据相似度列表计算断点位置。
    返回 [idx1, idx2, ...]，表示在 idx 和 idx+1 之间断开。
    """
    if not similarities:
        return []

    sims = np.array(similarities)
    if threshold_type == "percentile":
        threshold = np.percentile(sims, percentile)
    else:
        threshold = np.percentile(sims, percentile)

    breakpoints = []
    for i, sim in enumerate(sims):
        if sim < threshold:
            breakpoints.append(i)
    return breakpoints


class SemanticTextSplitter:
    """语义切分器：基于句子嵌入相似度检测主题边界。

    用法:
        from semantic_splitter import SemanticTextSplitter
        splitter = SemanticTextSplitter(embeddings, threshold_type="percentile", percentile=50)
        chunks = splitter.split_documents(docs)
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        threshold_type: Literal["percentile"] = "percentile",
        percentile: float = 50.0,
        min_chunk_size: int = 80,
        max_chunk_size: int = 800,
    ):
        """
        Args:
            embeddings: 用于计算句子嵌入的模型。
            threshold_type: 阈值模式，目前支持 "percentile"。
            percentile: 分位数，低于此分位数的相邻相似度视为断点（默认50=中位数）。
            min_chunk_size: 最短块长度（字符数），低于此则与相邻块合并。
            max_chunk_size: 最长块长度（字符数），超过则回退递归切分。
        """
        self.embeddings = embeddings
        self.threshold_type = threshold_type
        self.percentile = percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """对文档列表进行语义切分，返回新的 Document 列表。"""
        chunks: List[Document] = []
        for doc in documents:
            chunks.extend(self._split_document(doc))
        return chunks

    def _split_document(self, doc: Document) -> List[Document]:
        """对单个文档进行语义切分。"""
        text = doc.page_content.strip()
        if not text:
            return []

        sentences = _split_sentences(text)

        # 句子太少，直接作为一块
        if len(sentences) <= 1:
            if len(text) <= self.max_chunk_size:
                return [doc]
            else:
                return self._fallback_split(doc)

        # 如果总文本很短，不切
        if len(text) <= self.min_chunk_size * 2:
            return [doc]

        # 计算每个句子的嵌入
        embeddings_list = self.embeddings.embed_documents(sentences)
        embeds = np.array(embeddings_list)

        # 计算相邻句子相似度
        similarities = []
        for i in range(len(embeds) - 1):
            sim = _cosine_similarity(embeds[i], embeds[i + 1])
            similarities.append(sim)

        # 计算断点
        breakpoints = _compute_breakpoints(
            similarities,
            threshold_type=self.threshold_type,
            percentile=self.percentile,
        )

        # 如果没有断点，文本整体语义连续
        if not breakpoints:
            if len(text) <= self.max_chunk_size:
                return [doc]
            else:
                return self._fallback_split(doc)

        # 按断点组合句子
        chunk_sentences = []
        start = 0
        for bp in breakpoints:
            chunk = sentences[start:bp + 1]
            chunk_sentences.append(chunk)
            start = bp + 1
        if start < len(sentences):
            chunk_sentences.append(sentences[start:])

        # 后处理：合并过短的块
        merged = self._merge_short_chunks(chunk_sentences, doc.metadata)

        # 对超长块回退递归切分
        final_chunks: List[Document] = []
        for text_block, meta in merged:
            if len(text_block) > self.max_chunk_size:
                final_chunks.extend(self._fallback_split_single(text_block, meta))
            else:
                final_chunks.append(Document(page_content=text_block, metadata=meta.copy()))

        return final_chunks

    def _merge_short_chunks(
        self, chunk_sentences: List[List[str]], metadata: dict
    ) -> List[tuple]:
        """合并过短的块（< min_chunk_size）到相邻块。"""
        if len(chunk_sentences) <= 1:
            text = _join_sentences(chunk_sentences[0])
            return [(text, metadata)]

        texts = [_join_sentences(s) for s in chunk_sentences]
        merged = []
        pending = ""

        for t in texts:
            combined = pending + t
            if len(combined) < self.min_chunk_size:
                pending = combined
            else:
                if pending:
                    merged.append((pending, metadata))
                pending = t

        if pending:
            if merged and len(pending) < self.min_chunk_size:
                # 粘到最后一个块上
                merged[-1] = (merged[-1][0] + pending, metadata)
            else:
                merged.append((pending, metadata))

        return merged

    def _fallback_split(self, doc: Document) -> List[Document]:
        """超长或难以语义切分的文档回退到递归字符切分。"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=min(50, self.max_chunk_size // 10),
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        return splitter.split_documents([doc])

    def _fallback_split_single(self, text: str, metadata: dict) -> List[Document]:
        """对单个超长文本块回退递归切分。"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=min(50, self.max_chunk_size // 10),
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        return splitter.create_documents([text], metadatas=[metadata])
