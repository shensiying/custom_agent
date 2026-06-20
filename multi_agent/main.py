# main.py — 多 Agent 电商智能客服交互入口（LangGraph 编排 + 记忆系统）
from langchain_core.messages import HumanMessage, AIMessage
from supervisor import supervisor
from config import CURRENT_USER_ID


def main():
    user_id = CURRENT_USER_ID  # 用户隔离标识 = thread_id

    print("=" * 60)
    print("多 Agent 电商智能客服（LangGraph 编排 + 记忆系统）")
    print(f"  当前用户: {user_id}")
    print("  依赖服务: router(8002)  pre_service(8003)  after_service(8004)")
    print("  启动方式: bash start_all.sh")
    print("输入 'quit' 退出")
    print("=" * 60)
    print()

    # thread_id = user_id 实现用户隔离
    config = {"configurable": {"thread_id": user_id}}

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
            # 通过 LangGraph supervisor 编排
            # 只传入本轮用户消息，历史由 RedisSaver checkpoint 自动加载
            result = supervisor.invoke(
                {"messages": [HumanMessage(content=user_input)], "user_id": user_id},
                config=config,
            )

            # 提取最后一条 AI 回复
            response = ""
            for m in reversed(result["messages"]):
                if isinstance(m, AIMessage) and not m.content.startswith("[路由"):
                    response = m.content
                    break
            if not response:
                for m in reversed(result["messages"]):
                    if isinstance(m, AIMessage):
                        response = m.content
                        break

            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"处理出错: {e}\n")


if __name__ == "__main__":
    main()
