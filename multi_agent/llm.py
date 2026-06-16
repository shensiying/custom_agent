# llm.py — 共享 LLM 工厂
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL


def create_llm(temperature: float = 0.3) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def create_react_llm(tools: list, system_prompt: str = "", temperature: float = 0.3):
    """创建 ReAct Agent（LLM + 工具 + 系统提示）。"""
    llm = create_llm(temperature=temperature)
    prompt = SystemMessage(content=system_prompt) if system_prompt else None
    return create_react_agent(model=llm, tools=tools, prompt=prompt)
