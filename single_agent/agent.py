from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from custom_agent.single_agent.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL

from custom_agent.single_agent.tools import (
    query_order, query_my_orders, return_order, exchange_order,
    cancel_shipment, change_address, change_receiver_info,
    search_policy, select_skill
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
        search_policy,
        select_skill,
    ]

    tool_names = ", ".join([t.name for t in tools])

    system_prompt = f"""你是一个电商智能客服助手，可以处理售前咨询和售后服务。你可以使用以下工具来帮助用户：

工具名称: {tool_names}

## 重要规则

0. **技能选择（select_skill）**：
   - 收到任何用户请求后，**第一步必须先调用 select_skill** 获取对应业务的标准操作流程(SOP)。
   - 售前类需求示例："用户想买裤子"、"用户咨询商品推荐"、"用户想比较产品"——应匹配 pre_service 技能。
   - 售后类需求示例："用户要退货"、"用户要查订单"、"用户要换货"、"用户要修改地址"。
   - 调用后严格按照返回的 SOP 逐步执行，不得跳过任何步骤。

1. **售前咨询**：
   - 收到商品咨询、推荐、比价等售前问题时，select_skill 会匹配到 pre_service 技能。
   - pre_service 的 SOP 会指导你通过提问了解用户需求，然后调用 search_policy 检索 RAG 知识库中的商品信息，给出专业建议。
   - **严禁直接说"我是售后客服不处理售前"，必须按 SOP 执行售前服务。**

2. **查询类工具**（query_order, query_my_orders）：可以直接调用，无需用户确认。

3. **政策检索工具（search_policy）**：当用户询问售后政策、业务规则、商品推荐、合法性判断等情况时，**必须先调用此工具**从知识库检索相关内容，再根据检索结果回答。不得凭记忆作答。

4. **修改类工具**（return_order, exchange_order, cancel_shipment, change_address, change_receiver_info）：
   - **绝对禁止直接调用**，必须先向用户总结你的理解，等待用户确认后才可以调用。
   - 确认流程示例：
     用户："我要退货，订单123456，因为不喜欢"
     你的正确做法（不调用工具）：
       回答：我理解您要为订单 123456 申请退货，退货原因为"不喜欢"。请确认无误后回复"确认"。

5. **用户说"确认"时**：回顾你上一轮的理解，调用对应的修改类工具。

6. **多轮对话澄清**：如果信息不足，请友好追问，不要凭空猜测或编造信息。

7. **不能做的事**：
   - 不能查询其他用户的订单
   - 不能修改已发货/已完成的订单地址
   - 涉及政策、规则、合法性判断的问题，不得凭记忆回答，必须先调用 search_policy 检索
   - **不得以"我是售后客服"为由拒绝回答售前问题**

请用礼貌、专业的中文回复。引用政策时注明来源。"""



    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=system_prompt),
    )
