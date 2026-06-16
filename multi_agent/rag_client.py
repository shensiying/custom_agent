# rag_client.py — RAG 检索 HTTP 客户端
import requests
from config import RAG_BASE_URL, TOP_K


def search_rag(query: str, top_k: int = TOP_K) -> str:
    """调用 RAG 服务检索文档，返回格式化文本。"""
    try:
        resp = requests.get(
            f"{RAG_BASE_URL}/search",
            params={"query": query, "top_k": top_k},
            timeout=10,
        )
        if resp.status_code != 200:
            return f"[RAG 服务异常 HTTP {resp.status_code}]"
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return f"[未找到与「{query}」相关的文档]"

        lines = [f"共检索到 {len(results)} 条相关内容："]
        for i, r in enumerate(results, 1):
            src = r.get("metadata", {}).get("source_file", "未知来源")
            lines.append(f"\n[{i}] 来源: {src}\n{r['content'].strip()}")
        return "\n".join(lines)
    except requests.exceptions.ConnectionError:
        return "[RAG 服务未启动]"
    except Exception as e:
        return f"[检索异常: {e}]"
