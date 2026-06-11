from agent import build_agent
from memory import SummaryBufferMemory
from config import MEMORY_MAX_TOKEN_LIMIT, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage


def main():
    print("正在初始化电商售后客服 Agent（使用 DeepSeek 模型）...")
    print("正在连接 MySQL 数据库...")

    try:
        agent = build_agent()
        summary_llm = ChatDeepSeek(
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        memory = SummaryBufferMemory(summary_llm, max_context_tokens=MEMORY_MAX_TOKEN_LIMIT)
        print("=" * 60)
        print("客服 Agent 已启动！")
        print("支持业务：查询订单、修改收件信息、修改地址、退换货、拦截快递")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60)
        print()
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请检查：")
        print("1. DeepSeek API Key 是否正确配置")
        print("2. MySQL 数据库连接配置是否正确")
        return

    while True:
        try:
            user_input = input("用户: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if not user_input:
            continue

        try:
            # 1. 从记忆获取上下文（摘要 + 近期原文）
            context_msgs = memory.get_context_messages()

            # 2. 拼接：历史上下文 + 当前用户输入
            input_msgs = list(context_msgs)
            input_msgs.append(HumanMessage(content=user_input))

            # 3. 调用 Agent
            result = agent.invoke({"messages": input_msgs})
            assistant_msg = result["messages"][-1].content

            # 4. 保存本轮到记忆
            memory.add_turn(user_input, assistant_msg)

            print(f"Agent: {assistant_msg}\n")
        except Exception as e:
            print(f"处理出错: {e}\n")


if __name__ == "__main__":
    main()
