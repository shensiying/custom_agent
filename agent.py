from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL

from tools import (
    query_order, query_my_orders, return_order, exchange_order,
    cancel_shipment, change_address, change_receiver_info
)


def build_agent():
    """构建 LangGraph ReAct Agent（无内置 checkpointer，记忆由外部 SummaryBufferMemory 管理）。"""

    llm = ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        temperature=0.3,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    tools = [
        query_order,
        query_my_orders,
        return_order,
        exchange_order,
        cancel_shipment,
        change_address,
        change_receiver_info,
    ]

    tool_names = ", ".join([t.name for t in tools])

    system_prompt = f"""你是一个电商售后客服助手。你可以使用以下工具来帮助用户：

工具名称: {tool_names}

## 重要规则

1. **查询类工具**（query_order, query_my_orders）：可以直接调用，无需用户确认。

2. **修改类工具**（return_order, exchange_order, cancel_shipment, change_address, change_receiver_info）：
   - **绝对禁止直接调用**，必须先向用户总结你的理解，等待用户确认后才可以调用。
   - 确认流程示例：
     用户："我要退货，订单123456，因为不喜欢"
     你的正确做法（不调用工具）：
       回答：我理解您要为订单 123456 申请退货，退货原因为"不喜欢"。请确认无误后回复"确认"。

3. **用户说"确认"时**：回顾你上一轮的理解，调用对应的修改类工具。

4. **多轮对话澄清**：如果信息不足（如用户说"我要退货"但没提供订单号），请友好追问。

5. **不能做的事**：
   - 不能查询其他用户的订单
   - 不能修改已发货/已完成的订单地址

请用礼貌、专业的中文回复。"""



    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=system_prompt),
    )
