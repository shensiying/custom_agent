import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

from config import UPLOAD_DIR, TOP_K, BASE_DIR
from utils import get_file_md5, load_md5_cache, save_md5_cache
from rag_manager import process_and_add_document, search_similar, clear_vectorstore

app = FastAPI(title="RAG 文档服务")

# 模板引擎（用于展示上传页面）
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# 确保上传目录存在
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ==================== 离线模式：上传页面 ====================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cache = load_md5_cache()
    cache_display = "\n".join([f"{k} : {v}" for k, v in cache.items()])
    return templates.TemplateResponse(request, "index.html", {"cache_display": cache_display})

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档，自动加载、切片、MD5 去重、存入向量库
    """
    # 保存临时文件
    original_filename = file.filename
    safe_name = original_filename.replace("/", "_").replace("\\", "_")
    temp_path = Path(UPLOAD_DIR) / safe_name
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_md5 = get_file_md5(temp_path)

    # MD5 去重检查
    cache = load_md5_cache()
    for existing_name, existing_md5 in cache.items():
        if existing_md5 == file_md5:
            temp_path.unlink()
            raise HTTPException(status_code=400, detail=f"内容重复，已存在文件: {existing_name} (MD5: {file_md5})")

    try:
        chunk_count = process_and_add_document(temp_path, file_md5)
        # 更新缓存
        cache[original_filename] = file_md5
        save_md5_cache(cache)
        return {
            "status": "success",
            "filename": original_filename,
            "md5": file_md5,
            "chunks": chunk_count
        }
    except Exception as e:
        temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()

# ==================== 在线模式：检索接口 ====================
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K

class SearchResult(BaseModel):
    content: str
    metadata: dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.get("/search", response_model=SearchResponse)
async def search_get(query: str, top_k: int = TOP_K):
    docs = search_similar(query, top_k)
    results = [SearchResult(content=doc.page_content, metadata=doc.metadata) for doc in docs]
    return SearchResponse(query=query, results=results)

@app.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    docs = search_similar(request.query, request.top_k)
    results = [SearchResult(content=doc.page_content, metadata=doc.metadata) for doc in docs]
    return SearchResponse(query=request.query, results=results)

# ==================== 辅助接口 ====================
@app.get("/stats")
async def stats():
    from rag_manager import get_vectorstore
    count = get_vectorstore()._collection.count()
    return {"total_chunks": count}

@app.delete("/clear")
async def clear():
    clear_vectorstore()
    save_md5_cache({})
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)