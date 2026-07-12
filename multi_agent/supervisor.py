# supervisor.py — LangGraph 多 Agent 编排（HTTP + MemorySaver checkpoint + 长期记忆）
import asyncio
import json
from typing import Literal, AsyncGenerator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated

import requests
from llm import create_llm
from memory import (
    get_checkpointer,
    get_user_profile,
    upsert_user_profile,
    build_profile_context,
    extract_user_profile,
    manage_memory,
)


# 自定义消息 reducer：支持替换消息列表（记忆修剪）
def _messages_reducer(existing: list, new: list) -> list:
    """如果 new 是完整列表（来自 memory_prepare_node），则替换；
    否则作为增量追加（来自其他节点的 AIMessage 响应）。"""
    if not new:
        return existing
    # 如果 new 的第一个元素是含摘要的 SystemMessage，说明是替换操作
    if isinstance(new[0], SystemMessage) and new[0].content.startswith("[对话摘要]"):
        return new
    # 如果 new 长度 > 2（一次性替换整列表），视为替换
    if len(new) > 2:
        return new
    # 增量追加
    return existing + new

# 各 Agent 服务地址
ROUTER_URL = "http://127.0.0.1:8002"
PRE_SERVICE_URL = "http://127.0.0.1:8003"
AFTER_SERVICE_URL = "http://127.0.0.1:8004"


class MultiAgentState(TypedDict):
    messages: Annotated[list, _messages_reducer]
    route: str
    intent: str
    reasoning: str
    user_id: str
    profile_context: str


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

def memory_prepare_node(state: MultiAgentState) -> dict:
    """
    记忆准备节点（router 之前执行）：
    1. 管理短期记忆：检查是否需要概括总结
    2. 检索长期画像：构建 profile_context
    """
    messages = state.get("messages", [])
    user_id = state.get("user_id", "anonymous")

    # 1. 短期记忆管理
    managed = manage_memory(messages)

    # 2. 检索长期画像
    profile_ctx = build_profile_context(user_id)

    return {
        "messages": managed,
        "profile_context": profile_ctx,
    }


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
        return {"route": "general", "intent": "", "reasoning": str(e)}

    return {
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
    return {"messages": [AIMessage(content=response.content)]}


def pre_service_node(state: MultiAgentState) -> dict:
    """
    调用售前 Agent HTTP 服务。
    注入长期记忆中的用户画像作为 SystemMessage。
    """
    messages = state.get("messages", [])
    profile_ctx = state.get("profile_context", "")
    user_id = state.get("user_id", "anonymous")
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_text = m.content
            break

    serialized = _serialize_messages(messages)
    # 注入长期画像：在消息列表最前面插入 SystemMessage
    if profile_ctx:
        serialized.insert(0, {"role": "system", "content": profile_ctx})

    try:
        resp = requests.post(f"{PRE_SERVICE_URL}/chat", json={
            "user_input": user_text,
            "messages": serialized,
            "user_id": user_id,
        }, timeout=120)
        resp.raise_for_status()
        text = resp.json()["response"]
    except Exception as e:
        text = f"售前服务暂时不可用，请稍后重试。（{e}）"

    return {"messages": [AIMessage(content=text)]}


def after_service_node(state: MultiAgentState) -> dict:
    """调用售后 Agent HTTP 服务。"""
    messages = state.get("messages", [])
    user_id = state.get("user_id", "anonymous")
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
            "user_id": user_id,
        }, timeout=120)
        resp.raise_for_status()
        text = resp.json()["response"]
    except Exception as e:
        text = f"售后服务暂时不可用，请稍后重试。（{e}）"

    return {"messages": [AIMessage(content=text)]}


def clarify_node(state: MultiAgentState) -> dict:
    """引导用户重新表达需求。"""
    return {
        "messages": [AIMessage(content=(
            "抱歉呀，我没太理解您的意思～😅 请问您是想：\n\n"
            "1.  **了解或购买商品**（看看款式、尺码、价格什么的）\n"
            "2.  **处理已有订单**（退货、换货、查订单、改地址这类）\n\n"
            "简单跟我说一下就好，我来帮您搞定！"
        ))],
    }


def memory_update_node(state: MultiAgentState) -> dict:
    """
    记忆更新节点（业务完成后执行）：
    1. 从本轮对话中提取用户画像信息，更新长期记忆
    """
    messages = state.get("messages", [])
    user_id = state.get("user_id", "anonymous")

    # 提取并更新长期画像
    extract_user_profile(user_id, messages)

    return {}


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
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage)):
            role = "human" if isinstance(m, HumanMessage) else ("ai" if isinstance(m, AIMessage) else "system")
            result.append({"role": role, "content": m.content})
    return result


# ============================================================
# 构建 LangGraph
# ============================================================

def build_supervisor() -> StateGraph:
    graph = StateGraph(MultiAgentState)

    # 节点注册
    graph.add_node("memory_prepare", memory_prepare_node)
    graph.add_node("router", router_node)
    graph.add_node("general", general_node)
    graph.add_node("pre_service", pre_service_node)
    graph.add_node("after_service", after_service_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("memory_update", memory_update_node)

    # 入口：先做记忆准备，再路由
    graph.set_entry_point("memory_prepare")
    graph.add_edge("memory_prepare", "router")

    # 路由分发
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

    # 所有业务节点 → memory_update → END
    graph.add_edge("general", "memory_update")
    graph.add_edge("pre_service", "memory_update")
    graph.add_edge("after_service", "memory_update")
    graph.add_edge("clarify", "memory_update")
    graph.add_edge("memory_update", END)

    # 编译图，注入 MemorySaver checkpointer（可替换为 RedisSaver）
    return graph.compile(checkpointer=get_checkpointer())


# 全局编译实例
supervisor = build_supervisor()


# ============================================================
# 流式 Supervisor（SSE 事件生成器）
# ============================================================
async def astream_supervisor(user_input: str, user_id: str, messages: list) -> AsyncGenerator[dict, None]:
    """
    流式执行 supervisor，逐 token 产出 SSE 事件。

    策略：先用同步 invoke() 确定路由并生成完整响应（确保 checkpoint/记忆正常），
    再逐字符流式推送响应内容。对于 general 路由，本地再用 LLM stream 重生成一次
    以获取真正的 token 级流式体验。
    """
    config = {"configurable": {"thread_id": user_id}}
    input_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_id": user_id,
    }

    route = "general"
    intent = ""
    final_response = ""

    try:
        # 1. 先用同步 invoke 确定路由（确保记忆/checkpoint 正常更新）
        result = await asyncio.to_thread(supervisor.invoke, input_state, config)

        route = result.get("route", "general")
        intent = result.get("intent", "")

        # 提取完整 AI 响应
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessage) and not m.content.startswith("[路由"):
                final_response = m.content
                break

        # 2. 流式推送
        if route == "general" and final_response:
            # general 路由：用 LLM 重新流式生成，获得真正的 token 级流式
            yield {"type": "status", "content": ""}
            llm = create_llm(temperature=0.7)
            recent = [m for m in result.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))][-4:]
            full_msgs = [SystemMessage(content=GENERAL_PROMPT)] + recent
            streamed = ""
            async for chunk in llm.astream(full_msgs):
                if chunk.content:
                    streamed += chunk.content
                    yield {"type": "token", "content": chunk.content}
            final_response = streamed
        else:
            # 其他路由：逐字符流式推送 sync invoke 的响应
            if route == "pre_service":
                yield {"type": "status", "content": "正在为您查询商品信息…"}
            elif route == "after_service":
                yield {"type": "status", "content": "正在处理您的售后请求…"}

            for char in final_response:
                yield {"type": "token", "content": char}
                await asyncio.sleep(0.01)  # 小延迟模拟流式效果

    except Exception as e:
        yield {"type": "error", "content": f"服务异常: {str(e)}"}
        return

    yield {"type": "done", "route": route, "intent": intent, "response": final_response}
