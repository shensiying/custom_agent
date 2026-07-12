import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, project_root)

import importlib.util

rag_manager_path = os.path.join(project_root, 'RAG', 'rag_manager.py')
spec = importlib.util.spec_from_file_location("rag_manager", rag_manager_path)
rag_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_manager)
search_similar = rag_manager.search_similar
get_vectorstore = rag_manager.get_vectorstore

llm_path = os.path.join(project_root, 'multi_agent', 'llm.py')
spec = importlib.util.spec_from_file_location("llm", llm_path)
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)
create_llm = llm.create_llm

from typing import List, Dict
import numpy as np


def generate_test_cases() -> List[Dict]:
    """生成测试用例。根据RAG知识库内容动态生成。"""
    vs = get_vectorstore()
    collection = vs._collection
    data = collection.get(include=["documents"])
    
    docs = data["documents"] if data["documents"] else []
    
    if not docs:
        print("向量库为空，使用默认测试用例")
        return [
            {"query": "牛仔裤有哪些颜色？", "ground_truth": "牛仔裤常见颜色包括蓝色、黑色、灰色等"},
            {"query": "牛仔裤尺码怎么选？", "ground_truth": "应根据腰围、臀围和身高体重选择合适尺码"},
            {"query": "牛仔裤洗涤注意事项是什么？", "ground_truth": "深色牛仔裤第一次洗可能掉色，建议单独清洗，避免热水"},
            {"query": "牛仔裤的起源是什么？", "ground_truth": "牛仔裤起源于美国西部，最初为淘金工人设计"},
            {"query": "牛仔裤有哪些款式？", "ground_truth": "常见款式有直筒裤、修身裤、阔腿裤等"},
        ]
    
    test_cases = []
    for doc in docs[:10]:
        content = doc[:300]
        sentences = content.split("。")
        if len(sentences) >= 3:
            context = sentences[0].strip() + "。" + sentences[1].strip() + "。"
            question = sentences[2].strip()
            if question and len(question) > 5 and len(context) > 20:
                if "？" not in question:
                    question += "？"
                test_cases.append({
                    "query": question,
                    "ground_truth": context
                })
    
    if len(test_cases) < 5:
        test_cases = [
            {"query": "牛仔裤有哪些颜色？", "ground_truth": "牛仔裤常见颜色包括蓝色、黑色、灰色等"},
            {"query": "牛仔裤尺码怎么选？", "ground_truth": "应根据腰围、臀围和身高体重选择合适尺码"},
            {"query": "牛仔裤洗涤注意事项是什么？", "ground_truth": "深色牛仔裤第一次洗可能掉色，建议单独清洗，避免热水"},
            {"query": "牛仔裤的起源是什么？", "ground_truth": "牛仔裤起源于美国西部，最初为淘金工人设计"},
            {"query": "牛仔裤有哪些款式？", "ground_truth": "常见款式有直筒裤、修身裤、阔腿裤等"},
        ]
    
    return test_cases


def run_rag_pipeline(query: str) -> dict:
    """运行完整RAG管线：检索 → 生成回答"""
    retrieved_docs = search_similar(query, top_k=3)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    context_text = "\n\n".join(contexts)
    
    llm = create_llm(temperature=0.3)
    prompt = f"""基于以下文档内容回答问题：

文档内容：
{context_text}

问题：{query}

要求：
1. 只基于提供的文档内容回答
2. 如果文档中没有相关信息，明确说明"文档中未提及"
3. 回答要简洁准确
"""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    answer = response.content if hasattr(response, 'content') else str(response)
    
    return {
        "query": query,
        "contexts": contexts,
        "answer": answer
    }


def calculate_overlap_score(text1: str, text2: str) -> float:
    """计算两段文本的词汇重叠度"""
    words1 = [w for w in text1 if w.strip()]
    words2 = [w for w in text2 if w.strip()]
    
    if not words1 or not words2:
        return 0.0
    
    set1 = set(words1)
    set2 = set(words2)
    
    intersection = set1 & set2
    union = set1 | set2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def evaluate_retrieval_precision(query: str, contexts: List[str], ground_truth: str) -> float:
    """评估检索精确性：检索到的上下文与真实答案的相关性"""
    scores = []
    for ctx in contexts:
        score = calculate_overlap_score(ctx, ground_truth)
        scores.append(score)
    return np.mean(scores) if scores else 0.0


def evaluate_retrieval_recall(query: str, contexts: List[str], ground_truth: str) -> float:
    """评估检索召回率：真实答案中的关键信息是否被检索到"""
    all_context = "\n".join(contexts)
    return calculate_overlap_score(all_context, ground_truth)


def evaluate_answer_relevancy(query: str, answer: str) -> float:
    """评估回答与问题的相关性"""
    return calculate_overlap_score(query, answer)


def evaluate_faithfulness(answer: str, contexts: List[str]) -> float:
    """评估回答的忠实度：回答是否基于检索到的上下文"""
    all_context = "\n".join(contexts)
    return calculate_overlap_score(answer, all_context)


def evaluate_answer_accuracy(answer: str, ground_truth: str) -> float:
    """评估回答准确性：回答与真实答案的匹配度"""
    return calculate_overlap_score(answer, ground_truth)


def evaluate_with_custom_metrics(test_cases: List[Dict]) -> Dict:
    """使用自定义指标评估RAG系统"""
    results = {
        "query": [],
        "retrieval_precision": [],
        "retrieval_recall": [],
        "answer_relevancy": [],
        "faithfulness": [],
        "answer_accuracy": [],
        "context_count": [],
    }
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n正在处理测试用例 {i}/{len(test_cases)}")
        print(f"查询: {tc['query']}")
        print(f"真实答案: {tc['ground_truth'][:50]}...")
        
        try:
            rag_result = run_rag_pipeline(tc['query'])
            
            rp = evaluate_retrieval_precision(tc['query'], rag_result['contexts'], tc['ground_truth'])
            rr = evaluate_retrieval_recall(tc['query'], rag_result['contexts'], tc['ground_truth'])
            ar = evaluate_answer_relevancy(tc['query'], rag_result['answer'])
            fa = evaluate_faithfulness(rag_result['answer'], rag_result['contexts'])
            aa = evaluate_answer_accuracy(rag_result['answer'], tc['ground_truth'])
            
            results["query"].append(tc['query'])
            results["retrieval_precision"].append(rp)
            results["retrieval_recall"].append(rr)
            results["answer_relevancy"].append(ar)
            results["faithfulness"].append(fa)
            results["answer_accuracy"].append(aa)
            results["context_count"].append(len(rag_result['contexts']))
            
            print(f"检索到 {len(rag_result['contexts'])} 条上下文")
            print(f"生成回答: {rag_result['answer'][:100]}...")
            print(f"检索精确性: {rp:.2f}, 检索召回率: {rr:.2f}")
            print(f"回答相关性: {ar:.2f}, 忠实度: {fa:.2f}, 准确性: {aa:.2f}")
            
        except Exception as e:
            print(f"处理失败: {e}")
            continue
    
    return results


def print_evaluation_report(results: Dict) -> None:
    """打印评估报告"""
    if not results["query"]:
        print("没有成功处理的测试用例")
        return
    
    print("\n" + "="*60)
    print("RAG 系统评估报告")
    print("="*60)
    
    metrics = [
        ("检索精确性 (Retrieval Precision)", results["retrieval_precision"]),
        ("检索召回率 (Retrieval Recall)", results["retrieval_recall"]),
        ("回答相关性 (Answer Relevancy)", results["answer_relevancy"]),
        ("忠实度 (Faithfulness)", results["faithfulness"]),
        ("回答准确性 (Answer Accuracy)", results["answer_accuracy"]),
    ]
    
    for name, scores in metrics:
        avg_score = np.mean(scores) * 100
        print(f"\n{name}: {avg_score:.2f}%")
    
    avg_contexts = np.mean(results["context_count"])
    print(f"\n平均检索上下文数: {avg_contexts:.1f}")
    
    print("\n" + "-"*60)
    print("指标说明")
    print("-"*60)
    print("""
检索精确性: 检索到的上下文与真实答案的词汇重叠度（越高越好，理想值>70%）
检索召回率: 真实答案中的信息是否被检索到（越高越好，理想值>70%）
回答相关性: 回答与问题的相关性（越高越好，理想值>80%）
忠实度: 回答是否基于检索到的上下文（越高越好，理想值>85%）
回答准确性: 回答与真实答案的匹配度（越高越好，理想值>70%）
""")
    
    print("\n" + "-"*60)
    print("各测试用例详情")
    print("-"*60)
    for i, query in enumerate(results["query"]):
        print(f"\n测试用例 {i+1}: {query}")
        print(f"  检索精确性: {results['retrieval_precision'][i]*100:.1f}%")
        print(f"  检索召回率: {results['retrieval_recall'][i]*100:.1f}%")
        print(f"  回答相关性: {results['answer_relevancy'][i]*100:.1f}%")
        print(f"  忠实度: {results['faithfulness'][i]*100:.1f}%")
        print(f"  回答准确性: {results['answer_accuracy'][i]*100:.1f}%")


if __name__ == "__main__":
    print("="*60)
    print("RAG 系统评估")
    print("="*60)
    
    test_cases = generate_test_cases()
    print(f"\n共生成 {len(test_cases)} 个测试用例")
    
    results = evaluate_with_custom_metrics(test_cases)
    print_evaluation_report(results)