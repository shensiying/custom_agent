# supervisor.py — LangGraph 多 Agent 编排（通过 HTTP 调用独立部署的 3 个 Agent 服务）
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated

import requests
from llm import create_llm

# 各 Agent 服务地址
ROUTER_URL = "http://127.0.0.1:8002"
PRE_SERVICE_URL = "http://127.0.0.1:8003"
AFTER_SERVICE_URL = "http://127.0.0.1:8004"


class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    intent: str
    reasoning: str


GENERAL_PROMPT = """你是一个热情、温暖、有幽默感的电商客服「小智」。你不是冰冷的机器，而是一个乐于助人的朋友。

## 你的性格
- 温暖亲切，像朋友一样聊天
- 有适度的幽默感，可以适当用表情符号
- 对用户有耐心，善于倾听
- 即使在闲聊也自然地引导到"有什么可以帮您的？"

## 闲聊原则
- 用户打招呼 → 热情回应，自我介绍，问有什么可以帮忙
- 用户感谢 → 真诚表示不客气
- 用户夸奖 → 谦虚感谢
- 用户说再见 → 温暖告别
- 其他闲聊 → 友善回应后温柔引导到业务

请用自然的中文口语风格回复，不要用模板化的客服话术。"""


# ============================================================
# LangGraph 节点
# ============================================================

def router_node(state: MultiAgentState) -> dict:
    """调用 Router Agent HTTP 服务进行意图分析。"""
    messages = state.get("messages", [])
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_text = m.content
            break

    serialized = _serialize_messages(messages)
    try:
        resp = requests.post(f"{ROUTER_URL}/route", json={
            "user_input": user_text,
            "messages": serialized,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {**state, "route": "general", "intent": "", "reasoning": str(e)}

    return {
        **state,
        "route": data.get("route", "general"),
        "intent": data.get("intent", ""),
        "reasoning": data.get("reasoning", ""),
    }


def general_node(state: MultiAgentState) -> dict:
    """本地处理日常闲聊。"""
    messages = list(state.get("messages", []))
    llm = create_llm(temperature=0.7)
    recent = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-4:]
    full_msgs = [SystemMessage(content=GENERAL_PROMPT)] + recent
    response = llm.invoke(full_msgs)
    return {**state, "messages": [AIMessage(content=response.content)]}


def pre_service_node(state: MultiAgentState) -> dict:
    """调用售前 Agent HTTP 服务。"""
    messages = state.get("messages", [])
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_text = m.content
            break

    serialized = _serialize_messages(messages)
    try:
        resp = requests.post(f"{PRE_SERVICE_URL}/chat", json={
            "user_input": user_text,
            "messages": serialized,
        }, timeout=120)
        resp.raise_for_status()
        text = resp.json()["response"]
    except Exception as e:
        text = f"售前服务暂时不可用，请稍后重试。（{e}）"

    return {**state, "messages": [AIMessage(content=text)]}


def after_service_node(state: MultiAgentState) -> dict:
    """调用售后 Agent HTTP 服务。"""
    messages = state.get("messages", [])
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_text = m.content
            break

    serialized = _serialize_messages(messages)
    try:
        resp = requests.post(f"{AFTER_SERVICE_URL}/chat", json={
            "user_input": user_text,
            "messages": serialized,
        }, timeout=120)
        resp.raise_for_status()
        text = resp.json()["response"]
    except Exception as e:
        text = f"售后服务暂时不可用，请稍后重试。（{e}）"

    return {**state, "messages": [AIMessage(content=text)]}


def clarify_node(state: MultiAgentState) -> dict:
    """引导用户重新表达需求。"""
    return {
        **state,
        "messages": [AIMessage(content=(
            "抱歉呀，我没太理解您的意思～😅 请问您是想：\n\n"
            "1.  **了解或购买商品**（看看款式、尺码、价格什么的）\n"
            "2.  **处理已有订单**（退货、换货、查订单、改地址这类）\n\n"
            "简单跟我说一下就好，我来帮您搞定！"
        ))],
    }


# ============================================================
# 路由决策
# ============================================================

def route_decision(state: MultiAgentState) -> Literal["general", "pre_service", "after_service", "clarify"]:
    route = state.get("route", "general")
    if route not in ("general", "pre_service", "after_service", "clarify"):
        return "general"
    return route


def _serialize_messages(messages: list) -> list:
    """将 LangChain 消息对象序列化为 dict 列表，供 HTTP 传输。"""
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            result.append({"role": "ai", "content": m.content})
        elif isinstance(m, SystemMessage):
            result.append({"role": "system", "content": m.content})
    return result


# ============================================================
# 构建 LangGraph
# ============================================================

def build_supervisor() -> StateGraph:
    graph = StateGraph(MultiAgentState)

    graph.add_node("router", router_node)
    graph.add_node("general", general_node)
    graph.add_node("pre_service", pre_service_node)
    graph.add_node("after_service", after_service_node)
    graph.add_node("clarify", clarify_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "general": "general",
            "pre_service": "pre_service",
            "after_service": "after_service",
            "clarify": "clarify",
        }
    )

    graph.add_edge("general", END)
    graph.add_edge("pre_service", END)
    graph.add_edge("after_service", END)
    graph.add_edge("clarify", END)

    return graph.compile()


# 全局编译实例
supervisor = build_supervisor()
