# router_agent/service.py — 路由 Agent 独立服务 (FastAPI)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.tools import tool
from llm import create_react_llm
from rag_client import search_rag as _search_rag

app = FastAPI(title="Router Agent", description="意图分析与路由")

# ============================================================
# 关键词工具
# ============================================================
KEYWORD_MAP = {
    "介绍商品": "pre_service", "推荐": "pre_service", "有什么": "pre_service",
    "有没有": "pre_service", "想买": "pre_service", "想了解": "pre_service",
    "多少钱": "pre_service", "价格": "pre_service", "款式": "pre_service",
    "尺码": "pre_service", "折扣": "pre_service", "活动": "pre_service",
    "优惠": "pre_service", "新品": "pre_service", "店铺": "pre_service",
    "售前": "pre_service",
    "退货": "after_service", "退款": "after_service", "换货": "after_service",
    "取消订单": "after_service", "修改地址": "after_service", "改地址": "after_service",
    "修改电话": "after_service", "查订单": "after_service", "订单状态": "after_service",
    "物流": "after_service", "快递": "after_service", "拦截": "after_service",
    "售后": "after_service", "收件人": "after_service", "政策": "after_service",
    "规则": "after_service",
}


@tool
def keyword_match(user_input: str) -> str:
    """用预设关键词表快速检测用户输入中的售前/售后意图方向，返回命中的关键词和方向。"""
    hits = []
    directions = set()
    for kw, direction in KEYWORD_MAP.items():
        if kw in user_input:
            hits.append(kw)
            directions.add(direction)
    if len(directions) == 0:
        direction = "unclear"
    elif len(directions) == 1:
        direction = directions.pop()
    else:
        direction = "ambiguous"
    return (
        f'{{"hits": {hits}, "direction": "{direction}"}}\n'
        f'命中 {len(hits)} 个关键词，方向: {direction}。'
        f'请结合完整上下文做最终判断（如 "我不想退货" 不是售后请求）。'
    )


def search_rag(query: str) -> str:
    """RAG 检索包装，供 LLM tool-call 使用。"""
    return _search_rag(query)


ROUTER_PROMPT = """你是一个电商智能客服的对话路由专家。你的任务是将用户请求分类为以下四类：

- "general"       — 日常闲聊：打招呼、问候、感谢、告别、称赞、无聊寒暄等非业务类对话
- "pre_service"   — 售前咨询：商品推荐、款式介绍、价格库存、活动折扣等购物前问题
- "after_service" — 售后服务：退换货、订单查询、修改地址、拦截快递、取消订单、售后政策咨询等购物后问题
- "clarify"       — 无法判断，需要引导用户重新表达需求

## 判断原则
1. 先判断是否为日常闲聊（"你好"、"在吗"、"谢谢"、"再见"、"你真棒"等）→ general
2. 再判断售前/售后业务问题
3. 实在无法判断才走 clarify

## 判断流程（严格按顺序）

### 第一层：关键词快速判断
先用 keyword_match 工具检测。注意关键词只是参考，必须结合语义：
- "我不想退货" 虽有"退货"但意图是拒绝/其他
- "退货政策是什么" 虽有"退货"但意图是售后咨询 → after_service

### 第二层：RAG 文档辅助判断
不确定时调用 search_rag 检索。

### 第三层：引导用户
仍不确定 → clarify。

## 输出格式（严格遵守）
只输出一个 JSON：
{{"route": "<general|pre_service|after_service|clarify>", "intent": "简要意图描述", "reasoning": "判断依据"}}
"""


# ============================================================
# API Models
# ============================================================
class RouteRequest(BaseModel):
    user_input: str
    messages: list = []  # [{"role": "human"|"ai", "content": "..."}]


class RouteResponse(BaseModel):
    route: str
    intent: str
    reasoning: str


# ============================================================
# Routes
# ============================================================
@app.post("/route", response_model=RouteResponse)
def route_endpoint(req: RouteRequest):
    agent = create_react_llm(
        tools=[keyword_match, search_rag],
        system_prompt=ROUTER_PROMPT,
        temperature=0.1,
    )

    input_msgs = [{"role": "user", "content": f"请分析以下用户请求并输出路由 JSON：\n\n用户: {req.user_input}"}]
    result = agent.invoke({"messages": input_msgs})
    content = result["messages"][-1].content

    # 解析 JSON
    route = "general"
    intent = ""
    reasoning = ""
    try:
        text = content
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
        route = parsed.get("route", "general")
        intent = parsed.get("intent", "")
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, IndexError):
        user_text = req.user_input
        if len(user_text) <= 3:
            route, intent = "general", "日常闲聊"
        elif any(kw in user_text for kw in ["退货", "退款", "换货", "修改", "订单", "物流", "快递"]):
            if not any(kw in user_text for kw in ["不想", "不要", "不是"]):
                route, intent = "after_service", "售后服务"
        elif any(kw in user_text for kw in ["推荐", "想买", "有什么", "有没有", "多少钱", "款式", "尺码"]):
            route, intent = "pre_service", "售前咨询"
        else:
            route, intent = "general", "日常闲聊"

    return RouteResponse(route=route, intent=intent, reasoning=reasoning)


@app.get("/health")
def health():
    return {"status": "ok", "service": "router_agent"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
