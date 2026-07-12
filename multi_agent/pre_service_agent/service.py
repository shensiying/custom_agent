# pre_service_agent/service.py — 售前 Agent 独立服务 (FastAPI)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from llm import create_react_llm
from rag_client import search_rag

app = FastAPI(title="Pre-Service Agent", description="售前咨询：商品推荐、活动折扣等")

SYSTEM_PROMPT = """你是一个热情的电商售前客服专员，负责帮助用户了解商品、推荐商品、解答活动折扣等问题。

## 工作流程
1. 如果用户需求不够具体（如"想买裤子"），先通过提问了解：性别、身高体重、风格偏好、预算、使用场景等。
2. 调用 search_product_info 检索 RAG 知识库中的商品/活动信息。
3. 根据检索结果给出专业推荐和购买建议。
4. 如果检索结果不理想，可以换关键词重试，最多 3 次。
5. 确实无法解答的问题，建议用户联系人工客服。

## 注意事项
- 不涉及订单操作，不调用任何订单修改工具。
- 回答要具体，引用商品名称、价格、特点等信息。
- 语气热情但不过分夸张。
- 不要编造知识库中没有的商品信息。
"""


@tool
def search_product_info(query: str) -> str:
    """检索商品信息、活动说明、折扣政策等售前相关文档。"""
    return search_rag(query)


# ============================================================
# API Models
# ============================================================
class ChatRequest(BaseModel):
    user_input: str
    messages: list = []  # [{"role": "human"|"ai", "content": "..."}]


class ChatResponse(BaseModel):
    response: str


# ============================================================
# Routes
# ============================================================
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    agent = create_react_llm(tools=[search_product_info], system_prompt=SYSTEM_PROMPT, temperature=0.5)

    # 系统提示 + 长期画像（profile_context 可能已由 supervisor 注入为 system 消息）
    input_msgs = []
    for m in req.messages[-6:]:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "human":
            input_msgs.append(HumanMessage(content=content))
        elif role == "ai":
            input_msgs.append(AIMessage(content=content))
        elif role == "system":
            input_msgs.append(SystemMessage(content=content))
    input_msgs.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    input_msgs.append(HumanMessage(content=req.user_input))

    result = agent.invoke({"messages": input_msgs})
    last_msg = result["messages"][-1]
    return ChatResponse(response=last_msg.content)


@app.get("/health")
def health():
    return {"status": "ok", "service": "pre_service_agent"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
