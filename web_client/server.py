# server.py — Web client server (auth + chat proxy + static files)
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from auth import create_user, get_user_by_username, verify_password, create_token, decode_token

# --- Config -------------------------------------------------------
BASE_DIR = Path(__file__).parent
BACKEND_CHAT_URL = "http://127.0.0.1:8005/chat"
BACKEND_STREAM_URL = "http://127.0.0.1:8005/chat/stream"
RAG_BASE_URL = "http://127.0.0.1:8001"

app = FastAPI(title="E-Commerce Customer Service - Web Client")

# Static files & templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

security = HTTPBearer()


# --- Auth dependency ----------------------------------------------
def current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


# --- Pydantic models ----------------------------------------------
class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    user_input: str
    messages: list = []


# --- Page routes --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


# --- Auth API -----------------------------------------------------
@app.post("/api/auth/register")
def api_register(body: RegisterRequest):
    username = body.username.strip()
    password = body.password.strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    if len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少 2 个字符")
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="密码至少 4 个字符")

    user = create_user(username, password)
    if user is None:
        raise HTTPException(status_code=409, detail="用户名已存在")

    token = create_token(user)
    return {
        "token": token,
        "user": {"id": user["id"], "username": user["username"]},
    }


@app.post("/api/auth/login")
def api_login(body: LoginRequest):
    username = body.username.strip()
    password = body.password.strip()

    user = get_user_by_username(username)
    if user is None or not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_token(user)
    return {
        "token": token,
        "user": {"id": user["id"], "username": user["username"]},
    }


@app.get("/api/auth/me")
def api_me(user: dict = Depends(current_user)):
    return {"id": int(user["sub"]), "username": user["username"]}


# --- Chat API (proxy) ---------------------------------------------
@app.post("/api/chat")
async def api_chat(body: ChatRequest, user: dict = Depends(current_user)):
    """Forward chat request to the multi-agent backend."""
    if not body.user_input.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    payload = {
        "user_input": body.user_input,
        "user_id": f"web_{user['sub']}_{user['username']}",
        "messages": body.messages,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(BACKEND_CHAT_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="客服后端服务未启动，请稍后重试")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="客服响应超时，请重试")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"客服服务异常: {str(e)}")

    return {
        "route": data.get("route", "general"),
        "intent": data.get("intent", ""),
        "response": data.get("response", "抱歉，我没有理解您的问题。"),
        "messages": data.get("messages", []),
    }


# --- Chat Stream API (SSE proxy) -----------------------------------
@app.post("/api/chat/stream")
async def api_chat_stream(body: ChatRequest, user: dict = Depends(current_user)):
    """SSE 流式代理 — 将后端 SSE 流转发到前端。"""
    if not body.user_input.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    payload = {
        "user_input": body.user_input,
        "user_id": f"web_{user['sub']}_{user['username']}",
        "messages": body.messages,
    }

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", BACKEND_STREAM_URL, json=payload) as resp:
                    if resp.status_code != 200:
                        yield f"data: {{\"type\": \"error\", \"content\": \"后端服务异常 HTTP {resp.status_code}\"}}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            yield line + "\n"
        except httpx.ConnectError:
            yield 'data: {"type": "error", "content": "客服后端服务未启动，请稍后重试"}\n\n'
        except Exception as e:
            yield f'data: {{"type": "error", "content": "服务异常: {str(e)}"}}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- RAG Proxy (文档管理页面代理到 5000 端口) ---------------
@app.get("/rag")
async def rag_index(request: Request):
    """代理 RAG 文档上传管理页面。"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{RAG_BASE_URL}/")
            return HTMLResponse(content=resp.text, status_code=resp.status_code)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="RAG 服务未启动")


@app.post("/upload")
async def rag_upload(request: Request):
    """代理 RAG 文件上传（multipart/form-data）。"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 直接转发原始 multipart body
            body = await request.body()
            headers = {"content-type": request.headers.get("content-type", "")}
            resp = await client.post(f"{RAG_BASE_URL}/upload", content=body, headers=headers)
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="RAG 服务未启动")


@app.api_route("/rag/{path:path}", methods=["GET", "POST", "DELETE"])
async def rag_proxy(path: str, request: Request):
    """代理所有 RAG 子路径（/search, /stats, /clear 等）。"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            url = f"{RAG_BASE_URL}/{path}"
            if request.query_params:
                url += f"?{request.query_params}"

            if request.method == "GET":
                resp = await client.get(url)
            elif request.method == "DELETE":
                resp = await client.delete(url)
            else:
                body = await request.body()
                headers = {"content-type": request.headers.get("content-type", "application/json")}
                resp = await client.post(url, content=body, headers=headers)

            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type:
                return HTMLResponse(content=resp.text, status_code=resp.status_code)
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="RAG 服务未启动")


# --- Health check -------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "web_client"}


# --- Entry point --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
