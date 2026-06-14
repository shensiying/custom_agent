from typing import List
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# 中文文件常见编码，按优先级排列
_TEXT_ENCODINGS = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]

# 全局向量库（单例）
_embeddings = None
_vectorstore = None


def _load_text_file(file_path: str) -> List[Document]:
    """加载文本文件，自动检测编码。"""
    for enc in _TEXT_ENCODINGS:
        try:
            loader = TextLoader(file_path, encoding=enc)
            return loader.load()
        except (UnicodeDecodeError, RuntimeError):
            continue
    raise ValueError(f"无法识别文件编码，已尝试: {', '.join(_TEXT_ENCODINGS)}")


def get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
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
    加载、切片、生成 ID、添加到向量库
    返回切片数量
    """
    docs = load_document(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    # 过滤空内容切片（空白页、纯换行等）
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError("文档内容为空，无法生成有效切片。请确认文档包含可读文本。")
    ids = [f"{file_md5}_{i}" for i in range(len(chunks))]
    # 添加源文件名到元数据
    for chunk in chunks:
        chunk.metadata["source_file"] = file_path.name
    vs = get_vectorstore()
    vs.add_documents(chunks, ids=ids)
    return len(chunks)

def search_similar(query: str, top_k: int = None):
    """检索相似文档片段"""
    if top_k is None:
        from config import TOP_K
        top_k = TOP_K
    vs = get_vectorstore()
    return vs.similarity_search(query, k=top_k)

def clear_vectorstore():
    """清空整个向量库（危险操作）"""
    global _vectorstore
    vs = get_vectorstore()
    vs.delete_collection()
    # 重新创建
    _vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embeddings,
        collection_name="rag_docs"
    )
