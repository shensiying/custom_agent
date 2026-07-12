# main_api.py — 统一 API 网关（LangGraph 编排 + 记忆系统）
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from supervisor import supervisor, astream_supervisor
from config import CURRENT_USER_ID


app = FastAPI(title="Multi-Agent E-Commerce Customer Service", description="多 Agent 电商智能客服 API")


class ChatRequest(BaseModel):
    user_input: str
    user_id: str = ""
    messages: list = []


class ChatResponse(BaseModel):
    route: str
    intent: str
    response: str
    messages: list


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_id = req.user_id or CURRENT_USER_ID
    config = {"configurable": {"thread_id": user_id}}

    # 只传本轮消息，历史由 checkpoint 自动加载
    result = supervisor.invoke(
        {"messages": [HumanMessage(content=req.user_input)], "user_id": user_id},
        config=config,
    )

    route = result.get("route", "general")
    intent = result.get("intent", "")
    response = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage) and not m.content.startswith("[路由"):
            response = m.content
            break
    if not response:
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage):
                response = m.content
                break

    all_msgs = req.messages + [
        {"role": "human", "content": req.user_input},
        {"role": "ai", "content": response},
    ]

    return ChatResponse(route=route, intent=intent, response=response, messages=all_msgs)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE 流式聊天端点。
    事件格式: data: {"type": "token"|"status"|"done"|"error", ...}
    """
    user_id = req.user_id or CURRENT_USER_ID

    async def event_generator():
        async for event in astream_supervisor(
            user_input=req.user_input,
            user_id=user_id,
            messages=req.messages,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
