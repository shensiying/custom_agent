"""加载 skills/ 目录下的技能文件，提供匹配与列表功能。"""
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"


def load_all_skills() -> dict[str, dict]:
    """加载所有技能文件，返回 {skill_name: skill_dict}。"""
    skills = {}
    for fpath in sorted(SKILLS_DIR.glob("*.py")):
        if fpath.name.startswith("_"):
            continue
        text = fpath.read_text(encoding="utf-8")
        try:
            data = eval(text)
        except Exception:
            continue
        name = data.get("name", fpath.stem)
        skills[name] = data
    return skills


def _char_bigrams(s: str) -> set[str]:
    """提取字符级 bigram 集合，用于中文短文本匹配。"""
    s = s.lower().replace(" ", "")
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else {s}


def match_skill(need_description: str) -> dict | None:
    """
    根据用户需求描述匹配最合适的技能。
    使用字符 bigram 重叠 + 意图关键词命中评分，返回得分最高的 skill dict。
    """
    skills = load_all_skills()
    if not skills:
        return None

    query_bigrams = _char_bigrams(need_description)
    query_chars = set(need_description.lower().replace(" ", ""))

    # 意图关键词：这些词出现时，对应技能大幅加分
    intent_keywords = {
        "pre_service": ["买", "购买", "推荐", "咨询", "售前", "想买", "想了解", "有什么", "有没有", "价格", "多少钱", "款式", "尺码"],
        "return_order": ["退货", "退款", "不喜欢", "不要了", "退掉"],
        "exchange_order": ["换货", "换一个", "换尺码", "换颜色", "换一款"],
        "order_modify": ["改地址", "修改地址", "改收货", "改收件人", "改电话", "修改电话"],
        "order_query": ["查订单", "订单状态", "物流", "到哪了", "什么时候到", "查一下"],
    }

    best_score = -1
    best_skill = None

    for name, skill in skills.items():
        desc = skill.get("description", "")
        name_lower = skill.get("name", "").lower()
        target = desc.lower() + " " + name_lower
        target_bigrams = _char_bigrams(target)

        # bigram 重叠分
        overlap = len(query_bigrams & target_bigrams)
        score = overlap * 2

        # 字符命中加分
        target_chars = set(target)
        char_hit = len(query_chars & target_chars)
        score += char_hit

        # 意图关键词命中 → 大幅加分
        for kw in intent_keywords.get(name, []):
            if kw in need_description:
                score += 50

        # 查询完全包含在描述中 → 大幅加分
        if need_description in desc:
            score += 100

        if score > best_score:
            best_score = score
            best_skill = skill

    return best_skill if best_score > 0 else None


def list_skills_brief() -> str:
    """列出所有可用技能的名称和简介，供 Agent 参考。"""
    skills = load_all_skills()
    lines = []
    for name, skill in sorted(skills.items()):
        lines.append(f"- {name}: {skill.get('description', '无简介')}")
    return "\n".join(lines) if lines else "（暂无可用技能）"
