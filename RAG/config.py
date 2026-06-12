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

# HuggingFace 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)