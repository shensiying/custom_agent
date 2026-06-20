# memory.py — 多 Agent 记忆系统（PostgresSaver 短期 + PostgresStore 长期）
"""
短期记忆：
  - PostgresSaver checkpoint，thread_id = user_id，实现用户隔离
  - 保留近 N 轮对话，超出部分触发 LLM 概括总结
  - 概括后替换旧消息为 SystemMessage(summary=...)，清理冗余数据
  - 数据持久化到 PostgreSQL，进程重启不丢失

长期记忆：
  - PostgresStore 存储用户画像
  - namespace: ("user_profile",), key: user_id
  - 字段：身高、体重、鞋码、风格偏好、品牌偏好、消费范围等
  - 售前服务时检索并注入上下文
  - 数据持久化到 PostgreSQL，进程重启不丢失
"""
import json
import psycopg
from datetime import datetime, timezone
from typing import Optional

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage,
)

from config import (
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE,
    MAX_RECENT_ROUNDS, SUMMARIZE_TRIGGER, CLEAN_AFTER_SUMMARIZE,
)
from llm import create_llm

# ============================================================
# PostgreSQL 连接
# ============================================================
PG_CONN_STRING = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

_pg_conn_saver: Optional[psycopg.Connection] = None
_pg_conn_store: Optional[psycopg.Connection] = None
_pg_saver: Optional[PostgresSaver] = None
_pg_store: Optional[PostgresStore] = None


def _get_saver_conn() -> psycopg.Connection:
    """PostgresSaver 专用连接（匹配 from_conn_string 的参数）。"""
    global _pg_conn_saver
    if _pg_conn_saver is None or _pg_conn_saver.closed:
        _pg_conn_saver = psycopg.connect(
            PG_CONN_STRING,
            autocommit=True,
            prepare_threshold=0,
            row_factory=psycopg.rows.dict_row,
        )
    return _pg_conn_saver


def _get_store_conn() -> psycopg.Connection:
    """PostgresStore 专用连接（autocommit，确保 put 立即持久化）。"""
    global _pg_conn_store
    if _pg_conn_store is None or _pg_conn_store.closed:
        _pg_conn_store = psycopg.connect(
            PG_CONN_STRING,
            autocommit=True,
            prepare_threshold=0,
            row_factory=psycopg.rows.dict_row,
        )
    return _pg_conn_store


# ============================================================
# 短期记忆：PostgresSaver checkpoint（thread_id = user_id 实现隔离）
# ============================================================
USER_PROFILE_NS = ("user_profile",)


def get_checkpointer() -> PostgresSaver:
    """返回全局 PostgresSaver 实例（checkpoint 持久化到 PostgreSQL）。"""
    global _pg_saver
    if _pg_saver is None:
        _pg_saver = PostgresSaver(conn=_get_saver_conn())
        _pg_saver.setup()
    return _pg_saver


# ============================================================
# 长期记忆：PostgresStore（用户画像持久化到 PostgreSQL）
# ============================================================
def get_longterm_store() -> BaseStore:
    """返回全局 PostgresStore 实例。"""
    global _pg_store
    if _pg_store is None:
        _pg_store = PostgresStore(conn=_get_store_conn())
        _pg_store.setup()
    return _pg_store


def get_user_profile(user_id: str) -> dict:
    """查询用户长期画像。"""
    store = get_longterm_store()
    item = store.get(USER_PROFILE_NS, user_id)
    if item:
        return item.value
    return {}


def upsert_user_profile(user_id: str, updates: dict) -> dict:
    """更新用户长期画像（merge 模式）。"""
    store = get_longterm_store()
    current = get_user_profile(user_id)
    merged = {**current, **updates}
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    store.put(USER_PROFILE_NS, user_id, merged)
    return merged


# ============================================================
# 概括总结
# ============================================================
SUMMARY_PROMPT = """你是一个对话概括助手。请将以下对话历史概括为一段简洁的摘要，保留关键信息：

- 用户是谁（称呼、基本信息）
- 讨论了什么话题
- 做了哪些决策或操作
- 当前未完成的事情

用中文，不超过 200 字。"""


def summarize_messages(messages: list) -> str:
    """将旧消息概括为一段摘要文本。"""
    if not messages:
        return ""

    text_lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            text_lines.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage):
            text_lines.append(f"客服: {m.content}")

    if not text_lines:
        return ""

    conversation = "\n".join(text_lines[-20:])
    llm = create_llm(temperature=0.2)
    result = llm.invoke([
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=f"请概括以下对话：\n{conversation}"),
    ])
    return result.content.strip()


def manage_memory(messages: list) -> list:
    """
    管理短期记忆：
    1. 统计 human+ai 消息数
    2. 超过阈值则触发概括
    3. 用 SystemMessage(summary=...) 替换旧消息
    4. 只保留最近 CLEAN_AFTER_SUMMARIZE 条消息 + 概括
    """
    if not messages:
        return messages

    conversation_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    if len(conversation_msgs) <= SUMMARIZE_TRIGGER:
        return messages

    # 已有的概括
    existing_summary = ""
    for m in messages:
        if isinstance(m, SystemMessage) and m.content.startswith("[对话摘要]"):
            existing_summary = m.content
            break

    recent = conversation_msgs[-CLEAN_AFTER_SUMMARIZE:]
    old = conversation_msgs[:-CLEAN_AFTER_SUMMARIZE]

    summary_text = summarize_messages(old)

    full_summary = f"[对话摘要] {summary_text}"
    if existing_summary:
        full_summary = f"{existing_summary}\n[更新] {summary_text}"

    return [SystemMessage(content=full_summary)] + recent


# ============================================================
# 长期记忆提取
# ============================================================
EXTRACT_PROFILE_PROMPT = """你是一个用户画像提取助手。从对话中识别用户的基本信息和偏好，输出 JSON。

## 可提取的字段
- height_cm: 身高(cm)，数字
- weight_kg: 体重(kg)，数字
- gender: 性别，"male" 或 "female"
- shoe_size: 鞋码
- style_preferences: 风格偏好列表，如 ["简约", "运动", "商务"]
- favorite_brands: 喜欢的品牌列表
- budget_range: 消费预算范围，如 "100-300元"
- clothing_sizes: 衣服尺码偏好
- other_notes: 其他备注

## 规则
- 只提取对话中明确提到的信息，不要推测
- 如果用户说"140斤"，weight_kg 应为 70（单位转换）
- 如果本轮没有新信息，返回空 JSON: {}

## 输出格式（只输出 JSON，不要任何其他文字）
{{"height_cm": null, "weight_kg": 70, ...}}
"""


def extract_user_profile(user_id: str, messages: list) -> dict:
    """从最近对话中提取用户画像信息，更新长期记忆。"""
    if not messages:
        return {}

    recent = []
    for m in messages[-20:]:
        if isinstance(m, HumanMessage):
            recent.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage):
            recent.append(f"客服: {m.content}")

    if not recent:
        return {}

    llm = create_llm(temperature=0.1)
    result = llm.invoke([
        SystemMessage(content=EXTRACT_PROFILE_PROMPT),
        HumanMessage(content="\n".join(recent)),
    ])

    content = result.content.strip()
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        extracted = json.loads(content)
    except json.JSONDecodeError:
        return {}

    updates = {k: v for k, v in extracted.items() if v is not None and v != [] and v != ""}
    if not updates:
        return {}

    upsert_user_profile(user_id, updates)
    return updates


def build_profile_context(user_id: str) -> str:
    """构建用户画像上下文文本，注入到售前 Agent 的系统提示中。"""
    profile = get_user_profile(user_id)
    if not profile:
        return ""

    parts = []
    if profile.get("gender"):
        gender_label = "男士" if profile["gender"] == "male" else "女士"
        parts.append(f"性别: {gender_label}")
    if profile.get("height_cm"):
        parts.append(f"身高: {profile['height_cm']}cm")
    if profile.get("weight_kg"):
        parts.append(f"体重: {profile['weight_kg']}kg")
    if profile.get("shoe_size"):
        parts.append(f"鞋码: {profile['shoe_size']}")
    if profile.get("style_preferences"):
        parts.append(f"风格偏好: {', '.join(profile['style_preferences'])}")
    if profile.get("favorite_brands"):
        parts.append(f"喜欢品牌: {', '.join(profile['favorite_brands'])}")
    if profile.get("budget_range"):
        parts.append(f"消费预算: {profile['budget_range']}")
    if profile.get("clothing_sizes"):
        parts.append(f"尺码偏好: {profile['clothing_sizes']}")
    if profile.get("other_notes"):
        parts.append(f"备注: {profile['other_notes']}")

    if not parts:
        return ""

    return "## 用户画像（长期记忆）\n" + "\n".join(f"- {p}" for p in parts)
