from typing import List
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 回退用
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    SEMANTIC_SPLIT_ENABLED, SEMANTIC_SPLIT_PERCENTILE,
    SEMANTIC_MAX_CHUNK_SIZE, SEMANTIC_MIN_CHUNK_SIZE,
    ENABLE_HYBRID_RETRIEVAL, ENABLE_RERANK,
    HYBRID_DENSE_K, HYBRID_SPARSE_K, DENSE_WEIGHT, SPARSE_WEIGHT,
    RERANK_CANDIDATE_K,
)
from semantic_splitter import SemanticTextSplitter
from retrievers import HybridRetriever, load_or_build_bm25, build_bm25_from_chroma
from reranker import rerank

# 中文文件常见编码，按优先级排列
_TEXT_ENCODINGS = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]

# 全局向量库（单例）
_embeddings = None
_vectorstore = None
_hybrid_retriever = None


_semantic_splitter = None  # 语义切分器（复用 embeddings）


def _load_text_file(file_path: str) -> List[Document]:
    """加载文本文件，自动检测编码。"""
    for enc in _TEXT_ENCODINGS:
        try:
            loader = TextLoader(file_path, encoding=enc)
            return loader.load()
        except (UnicodeDecodeError, RuntimeError):
            continue
    raise ValueError(f"无法识别文件编码，已尝试: {', '.join(_TEXT_ENCODINGS)}")

# 获取向量库（单例）
def get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'local_files_only': True},
        )
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_embeddings,
            collection_name="rag_docs"
        )
    return _vectorstore


def load_document(file_path: str) -> List[Document]:
    """根据扩展名加载文档，返回 Document 列表"""
    ext = file_path.suffix.lower()
    if ext in (".txt", ".md"):
        return _load_text_file(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
    return loader.load()


def process_and_add_document(file_path: str, file_md5: str) -> int:
    """
    加载、语义切分、生成 ID、添加到向量库
    返回切片数量
    """
    docs = load_document(file_path)

    # ===== 语义切分（默认启用） =====
    global _semantic_splitter
    if SEMANTIC_SPLIT_ENABLED:
        if _semantic_splitter is None:
            embeddings = get_vectorstore().embeddings
            _semantic_splitter = SemanticTextSplitter(
                embeddings,
                threshold_type="percentile",
                percentile=SEMANTIC_SPLIT_PERCENTILE,
                max_chunk_size=SEMANTIC_MAX_CHUNK_SIZE,
                min_chunk_size=SEMANTIC_MIN_CHUNK_SIZE,
            )
        chunks = _semantic_splitter.split_documents(docs)
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        chunks = splitter.split_documents(docs)

    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError("文档内容为空，无法生成有效切片。请确认文档包含可读文本。")
    ids = [f"{file_md5}_{i}" for i in range(len(chunks))]
    for chunk in chunks:
        chunk.metadata["source_file"] = file_path.name
    vs = get_vectorstore()
    vs.add_documents(chunks, ids=ids)
    # 文档变更后重建 BM25 索引
    _rebuild_bm25(vs)
    return len(chunks)


def search_similar(query: str, top_k: int = None):
    """检索相似文档片段（支持混合检索 + 重排序）"""
    if top_k is None:
        top_k = TOP_K

    if ENABLE_HYBRID_RETRIEVAL or ENABLE_RERANK:
        return _search_hybrid(query, top_k)

    # 原始单路稠密检索
    vs = get_vectorstore()
    return vs.similarity_search(query, k=top_k)


def _search_hybrid(query: str, top_k: int):
    """混合检索 + 可选重排序管线"""
    vs = get_vectorstore()
    global _hybrid_retriever
    if _hybrid_retriever is None:
        bm25 = load_or_build_bm25(vs)
        _hybrid_retriever = HybridRetriever(
            vs, bm25,
            dense_weight=DENSE_WEIGHT, sparse_weight=SPARSE_WEIGHT,
            dense_k=HYBRID_DENSE_K, sparse_k=HYBRID_SPARSE_K,
        )

    candidate_k = RERANK_CANDIDATE_K if ENABLE_RERANK else top_k
    candidates = _hybrid_retriever.search(query, top_k=candidate_k)

    if ENABLE_RERANK and len(candidates) > 1:
        return rerank(query, candidates, top_k=top_k)

    return candidates[:top_k]


def _rebuild_bm25(vs=None):
    """重建 BM25 索引（文档变更后调用）。"""
    global _hybrid_retriever
    if vs is None:
        vs = get_vectorstore()
    build_bm25_from_chroma(vs)
    _hybrid_retriever = None  # 下次搜索时懒重建


def clear_vectorstore():
    """清空整个向量库（危险操作）"""
    global _vectorstore, _hybrid_retriever
    vs = get_vectorstore()
    vs.delete_collection()
    _vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embeddings,
        collection_name="rag_docs"
    )
    _hybrid_retriever = None
