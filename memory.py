"""记忆模块：带自动总结的缓冲记忆。

- 用 tiktoken 精确计算 token 数
- 控制总上下文（摘要 + 近期原文）不超过上限
- 按 token 数动态保留近期窗口，切分始终对齐轮次边界
- 避免频繁触发总结（总结后留有缓冲余量）
"""

import tiktoken
from langchain_core.messages import HumanMessage, AIMessage


class SummaryBufferMemory:
    """近期原文 + 远期摘要 的缓冲记忆。

    参数:
        llm: 用于生成摘要的 LLM
        max_context_tokens: 发送给 Agent 的上下文总 token 预算，默认 4000
        recent_ratio: 近期原文占预算的比例，默认 0.7
    """

    def __init__(self, llm, max_context_tokens: int = 4000, recent_ratio: float = 0.7):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.summary: str = ""                # 旧对话摘要
        self.recent: list = []                # 近期原文 [{role, content}, ...]

        # tokenizer: cl100k_base 兼容 ChatGPT / DeepSeek / Qwen 等主流模型
        self._enc = tiktoken.get_encoding("cl100k_base")

        # 触发压缩的阈值：recent 的 token 数超过这个就压缩
        self._compact_threshold = int(max_context_tokens * recent_ratio)

        # 压缩后 recent 的目标值（留 30% 缓冲，避免刚压完又触发）
        self._compact_target = int(self._compact_threshold * 0.7)

    # ---- 公开接口 ----

    def add_turn(self, user_msg: str, assistant_msg: str):
        """保存一轮对话，必要时触发总结。"""
        self.recent.append({"role": "user", "content": user_msg})
        self.recent.append({"role": "assistant", "content": assistant_msg})

        if self._tokens_of(self.recent) > self._compact_threshold:
            self._compact()

    def get_context_messages(self) -> list:
        """构造发给 Agent 的消息列表。

        顺序: [摘要(如果有)] + [近期原文]
        没有历史则返回空列表。
        """
        # 兜底检查：摘要膨胀也可能导致超限
        while True:
            total = self._tokens_of([self.summary] if self.summary else []) + self._tokens_of(self.recent)
            if total <= self.max_context_tokens:
                break
            if not self.recent:
                # 摘要本身就超了 → 截断摘要
                self.summary = self._llm_summarize(
                    f"请用两句话压缩以下摘要：\n\n{self.summary}"
                )
                continue
            self._compact()

        result = []
        if self.summary:
            result.append(HumanMessage(
                content=f"[历史对话摘要]\n{self.summary}"
            ))
        for item in self.recent:
            if item["role"] == "user":
                result.append(HumanMessage(content=item["content"]))
            else:
                result.append(AIMessage(content=item["content"]))
        return result

    # ---- 内部 ----

    def _compact(self):
        """按 token 数从队尾向前保留，切分沿轮次（turn）边界对齐。

        至少保留 1 轮对话（2 条消息），保留到 token 数 ≤ _compact_target 为止。
        """
        if not self.recent:
            return

        # 从队尾向前按"轮"（每轮 2 条消息）累积 token，找到分界点
        turns = [(self.recent[i], self.recent[i + 1]) for i in range(0, len(self.recent), 2)]
        kept_turns = 0
        kept_tokens = 0
        min_turns = 1  # 至少保留 1 轮

        for user_msg, assistant_msg in reversed(turns):
            turn_tokens = (
                self._count_tokens(user_msg["content"]) +
                self._count_tokens(assistant_msg["content"])
            )
            if kept_turns >= min_turns and kept_tokens + turn_tokens > self._compact_target:
                break
            kept_tokens += turn_tokens
            kept_turns += 1

        # 全部保留也不超 → 不需要压缩
        if kept_turns == len(turns):
            return

        # 切分
        old_count = (len(turns) - kept_turns) * 2
        old_messages = self.recent[:old_count]
        self.recent = self.recent[old_count:]

        # 拼成对话文本
        old_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in old_messages
        )

        # 一次性总结（已有摘要则合并）
        if self.summary:
            prompt = (
                f"已有摘要：\n{self.summary}\n\n"
                f"新对话：\n{old_text}\n\n"
                "请将以上已有摘要和新对话合并成一份简洁完整的摘要，保留所有关键信息（订单号、操作、结果）："
            )
        else:
            prompt = (
                f"{old_text}\n\n"
                "请总结以上对话要点，保留关键信息（订单号、操作、结果）："
            )

        self.summary = self._llm_summarize(prompt)

    def _llm_summarize(self, text: str) -> str:
        """用 LLM 压缩文本。失败时降级为截断。"""
        try:
            msg = HumanMessage(content=text)
            resp = self.llm.invoke([msg])
            return resp.content.strip()
        except Exception:
            return text[:300] + "..."

    # ---- token 工具 ----

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _tokens_of(self, items: list) -> int:
        total = 0
        for item in items:
            if isinstance(item, str):
                total += self._count_tokens(item)
            elif isinstance(item, dict):
                total += self._count_tokens(item.get("content", ""))
        return total
