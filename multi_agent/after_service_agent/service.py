# after_service_agent/service.py — 售后 Agent 独立服务 (FastAPI)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from llm import create_react_llm
from database import get_order, get_orders, update_order
from rag_client import search_rag
from skills_loader import match_skill, list_skills_brief

app = FastAPI(title="After-Service Agent", description="售后服务：退换货、订单查询、修改地址等")

# ============================================================
# 工具函数（与单 agent 保持一致，但包装为 LangChain tool）
# ============================================================
from langchain_core.tools import tool


def _can_return(order: dict) -> tuple:
    status = order.get("status", "")
    if status not in ["completed", "delivered"]:
        return False, f"订单状态为 {status}，无法退货。"
    created_at = datetime.fromisoformat(order["created_at"]) if isinstance(order["created_at"], str) else order["created_at"]
    if (datetime.now() - created_at).days > 7:
        return False, "订单已超过7天退货期。"
    return True, "符合退货条件"


def _can_exchange(order: dict) -> tuple:
    status = order.get("status", "")
    if status not in ["completed", "delivered", "shipped"]:
        return False, f"订单状态为 {status}，无法换货。"
    created_at = datetime.fromisoformat(order["created_at"]) if isinstance(order["created_at"], str) else order["created_at"]
    if (datetime.now() - created_at).days > 15:
        return False, "订单已超过15天换货期。"
    return True, "符合换货条件"


def _can_modify(order: dict) -> tuple:
    if order.get("status") in ["pending", "paid"]:
        return True, ""
    return False, f"订单已 {order.get('status')}，无法修改。"


def _can_cancel(order: dict) -> tuple:
    if order.get("status") == "cancelled":
        return False, "订单已取消。"
    if order.get("logistics_status") == "delivered":
        return False, "订单已签收，无法拦截。"
    return True, "可以拦截/取消。"


@tool
def query_order(order_id: str) -> str:
    """查询订单详情。参数为6位数字订单号。"""
    order = get_order(order_id)
    if not order:
        return f"未找到订单 {order_id}。"
    return (
        f"订单{order_id}：状态={order['status']}，金额={order['amount']}元，"
        f"物流={order.get('logistics_status','N/A')}，地址={order['shipping_address']}，"
        f"收件人={order.get('receiver_name','')} {order.get('receiver_phone','')}，"
        f"创建于{order['created_at']}"
    )


@tool
def return_order(order_id: str, reason: str = "未提供") -> str:
    """申请退货。只有 completed/delivered 且 ≤7 天的订单可退货。"""
    order = get_order(order_id)
    if not order: return f"未找到订单 {order_id}。"
    ok, msg = _can_return(order)
    if not ok: return msg
    if update_order(order_id, {"status": "returning"}):
        return f"退货成功！订单 {order_id} 已进入退货流程。退款将在3个工作日内原路返回。"
    return "退货失败，请稍后重试。"


@tool
def query_my_orders(query: str = "") -> str:
    """查询当前用户所有订单。"""
    orders = get_orders()
    if not orders: return "您暂无订单记录。"
    lines = [f"订单号: {o['order_id']} | 状态: {o['status']} | 金额: {o['amount']}元 | 日期: {o['created_at']}" for o in orders]
    return "\n".join(lines)


@tool
def exchange_order(order_id: str, reason: str = "") -> str:
    """为指定订单申请换货。"""
    order = get_order(order_id)
    if not order: return f"未找到订单 {order_id}。"
    ok, msg = _can_exchange(order)
    if not ok: return msg
    if update_order(order_id, {"status": "exchanging"}):
        extra = f"（原因：{reason}）" if reason else ""
        return f"换货申请成功！订单 {order_id} 已进入换货流程{extra}。"
    return "换货失败，请稍后重试。"


@tool
def cancel_shipment(order_id: str) -> str:
    """拦截/取消订单快递。"""
    order = get_order(order_id)
    if not order: return f"未找到订单 {order_id}。"
    ok, msg = _can_cancel(order)
    if not ok: return msg
    new_status = "cancelled" if order["status"] == "pending" else "cancelling"
    if update_order(order_id, {"status": new_status}):
        return f"操作成功！订单 {order_id} 状态已更新为 {new_status}。"
    return "操作失败，请稍后重试。"


@tool
def change_address(order_id: str, new_address: str) -> str:
    """修改订单收货地址。"""
    order = get_order(order_id)
    if not order: return f"未找到订单 {order_id}。"
    ok, msg = _can_modify(order)
    if not ok: return msg
    if update_order(order_id, {"shipping_address": new_address}):
        return f"地址修改成功！订单 {order_id} 的新地址为：{new_address}"
    return "修改失败，请稍后重试。"


@tool
def change_receiver_info(order_id: str, name: str = "", phone: str = "") -> str:
    """修改收件人信息（姓名/电话）。"""
    order = get_order(order_id)
    if not order: return f"未找到订单 {order_id}。"
    ok, msg = _can_modify(order)
    if not ok: return msg
    updates = {}
    if name: updates["receiver_name"] = name
    if phone: updates["receiver_phone"] = phone
    if not updates: return "请至少提供姓名或电话。"
    if update_order(order_id, updates):
        return f"收件人信息修改成功！{', '.join(f'{k}: {v}' for k, v in updates.items())}"
    return "修改失败，请稍后重试。"


@tool
def search_policy(query: str) -> str:
    """检索售后政策、商品信息、业务规则等 RAG 知识库文档。"""
    return search_rag(query)


@tool
def select_skill(need_description: str) -> str:
    """
    根据用户需求描述匹配售后业务技能（SOP）。
    必须在处理具体请求前调用，然后严格按照返回的 SOP 执行。
    """
    skill = match_skill(need_description)
    if skill is None:
        available = list_skills_brief()
        return f"未找到匹配技能。\n可用技能：\n{available}"
    return (
        f"技能名称: {skill['name']}\n"
        f"技能描述: {skill['description']}\n"
        f"{'='*50}\n"
        f"标准操作流程(SOP)：\n{skill['content']}\n"
        f"{'='*50}\n"
        f"请严格按照上述流程逐步执行，不得跳过任何步骤。"
    )


TOOLS = [
    query_order, query_my_orders, return_order, exchange_order,
    cancel_shipment, change_address, change_receiver_info,
    search_policy, select_skill,
]

BASE_PROMPT = """你是一个专业的电商售后客服专员。你可以使用工具来处理退换货、订单查询、修改地址、拦截快递等售后问题。

## 核心规则
1. **技能选择（select_skill）**：处理任何售后请求前，第一步必须先调用 select_skill 获取 SOP，然后严格按 SOP 执行。
2. **修改类工具**（return_order, exchange_order, cancel_shipment, change_address, change_receiver_info）：必须先向用户总结，等待确认后才可调用。
3. **查询类工具**（query_order, query_my_orders）：可直接调用。
4. **政策检索**（search_policy）：涉及政策、规则时必须先检索 RAG。
5. **多轮澄清**：信息不足时友好追问。

请用礼貌、专业的中文回复。"""


# ============================================================
# API Models
# ============================================================
class ChatRequest(BaseModel):
    user_input: str
    messages: list = []


class ChatResponse(BaseModel):
    response: str


# ============================================================
# Routes
# ============================================================
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    agent = create_react_llm(tools=TOOLS, system_prompt=BASE_PROMPT, temperature=0.3)

    input_msgs = [SystemMessage(content=BASE_PROMPT)]
    for m in req.messages[-8:]:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "human":
            input_msgs.append(HumanMessage(content=content))
        elif role == "ai":
            input_msgs.append(AIMessage(content=content))
    input_msgs.append(HumanMessage(content=req.user_input))

    result = agent.invoke({"messages": input_msgs})
    last_msg = result["messages"][-1]
    return ChatResponse(response=last_msg.content)


@app.get("/health")
def health():
    return {"status": "ok", "service": "after_service_agent"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
