import hashlib
import json
from pathlib import Path
from config import MD5_CACHE_FILE

MD5_CACHE_PATH = Path(MD5_CACHE_FILE)

def get_file_md5(file_path) -> str:
    """计算文件的 MD5 值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_md5_cache():
    """加载 MD5 缓存（记录已处理文件的 MD5）"""
    if MD5_CACHE_PATH.exists():
        with open(MD5_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_md5_cache(cache):
    """保存 MD5 缓存"""
    with open(MD5_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)