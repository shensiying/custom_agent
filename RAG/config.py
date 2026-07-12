import os

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "documents_upload")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
MD5_CACHE_FILE = os.path.join(BASE_DIR, "md5_cache.json")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# 中文嵌入模型，第一次运行会自动下载（约100MB）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 3

# 语义切分配置
SEMANTIC_SPLIT_ENABLED = True      # 是否启用语义切分（关闭则回退递归字符切分）
SEMANTIC_SPLIT_PERCENTILE = 50.0   # 相似度分位数阈值（越低越保守，断点越少）
SEMANTIC_MAX_CHUNK_SIZE = 800      # 单个块最大字符数（超长回退递归切分）
SEMANTIC_MIN_CHUNK_SIZE = 80       # 单个块最小字符数（过短则合并）

# 多路召回 + 重排序配置
ENABLE_HYBRID_RETRIEVAL = True   # 开启 BM25 + 稠密混合检索
ENABLE_RERANK = True             # 开启 Cross-Encoder 重排序
HYBRID_DENSE_K = 10              # 稠密检索召回数（用于混合融合）
HYBRID_SPARSE_K = 10             # 稀疏检索召回数（用于混合融合）
DENSE_WEIGHT = 0.6               # 稠密向量权重
SPARSE_WEIGHT = 0.4              # BM25 权重
RERANK_CANDIDATE_K = 20          # 送入重排序的候选数
RERANKER_MODEL = "BAAI/bge-reranker-base"

# HuggingFace 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 离线模式：优先使用本地缓存，避免网络不可达时报错
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)