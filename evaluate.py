"""
evaluate.py — RAG 全链路评估脚本

衡量 4 个核心指标：
  1. Context Recall    — 检索找到了多少应该找到的文档
  2. Faithfulness      — 生成的回复中有没有编造检索文档之外的内容
  3. Answer Relevance  — 回复是否真的回答了客户问题
  4. Negative Rejection — 知识库没有的内容是否正确拒绝回答

用法：
  python evaluate.py              # 运行评估
  python evaluate.py --verbose    # 打印每条详情
  python evaluate.py --no-llm     # 只测检索，跳过 LLM 生成评估（更快）

依赖：DeepSeek API（.env 里的 DEEPSEEK_API_KEY）
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# 确保 rag 能 import
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

# ============================================================
# 测试集 — 每个用例：问题 + 期望检索到的关键词 + 参考要点
# ============================================================
TEST_CASES = [
    # ── 检索能力测试 ──
    {
        "id": "R01",
        "question": "圆脸戴什么珍珠好看",
        "expected_keywords": ["圆脸", "V字", "长款", "显脸小"],
        "reference_facts": ["圆脸推荐V字型或长款项链", "避免短链或卡脖子根的款式"],
    },
    {
        "id": "R02",
        "question": "500块能买什么珍珠",
        "expected_keywords": ["500", "淡水", "项链", "耳钉"],
        "reference_facts": ["500左右可以买淡水珍珠项链或耳钉", "这个价位不用追求海水珠"],
    },
    {
        "id": "R03",
        "question": "送妈妈生日礼物选什么珍珠",
        "expected_keywords": ["妈妈", "生日", "大溪地", "南洋金珠", "Akoya"],
        "reference_facts": ["送妈妈推荐大溪地或南洋金珠", "或者Akoya珍珠项链"],
    },
    {
        "id": "R04",
        "question": "淡水珍珠和海水珍珠有什么区别",
        "expected_keywords": ["淡水", "海水", "光泽", "性价比"],
        "reference_facts": ["海水珍珠光泽更锐利更亮但价格贵", "淡水珍珠光泽温润性价比高"],
    },
    {
        "id": "R05",
        "question": "珍珠怎么判断好不好",
        "expected_keywords": ["光泽", "圆度", "瑕疵", "大小"],
        "reference_facts": ["看光泽圆度瑕疵大小四个方面", "光泽最重要"],
    },
    {
        "id": "R06",
        "question": "自己日常戴买什么样的珍珠合适",
        "expected_keywords": ["日常", "百搭", "通勤"],
        "reference_facts": ["推荐日常百搭的款式", "不要太夸张"],
    },
    {
        "id": "R07",
        "question": "20多岁女生戴什么珍珠不老气",
        "expected_keywords": ["年轻", "小直径", "锁骨链", "不老气"],
        "reference_facts": ["推荐小直径珍珠锁骨链", "太大的会显老气"],
    },
    {
        "id": "R08",
        "question": "珍珠怎么保养",
        "expected_keywords": ["保养", "软布", "香水", "绒布袋", "先摘"],
        "reference_facts": ["戴完用软布擦", "别碰香水化妆品", "单独放绒布袋"],
    },
    # ── 负样本（知识库外，应拒绝回答） ──
    {
        "id": "N01",
        "question": "你们家今天珍珠打几折",
        "expected_keywords": [],
        "reference_facts": [],
        "is_negative": True,  # 库存/折扣是动态信息，应回"我帮你查一下"
    },
    {
        "id": "N02",
        "question": "帮我查一下顺丰快递到哪了",
        "expected_keywords": [],
        "reference_facts": [],
        "is_negative": True,  # 快递查询不在知识库范围内
    },
    # ── 模糊查询（测试 Query 改写效果） ──
    {
        "id": "F01",
        "question": "好看的",
        "expected_keywords": ["推荐", "珍珠"],
        "reference_facts": ["应该给出珍珠推荐而非拒绝"],
    },
    {
        "id": "F02",
        "question": "贵不贵",
        "expected_keywords": ["价格", "预算", "便宜"],
        "reference_facts": ["应该解释珍珠价位范围而非拒绝"],
    },
]

# ============================================================
# LLM Judge — 用 DeepSeek 做 Faithfulness & Relevance 评分
# ============================================================
FAITHFULNESS_PROMPT = """你是严格的事实核查员。请判断以下 AI 回复中的每一条陈述是否能在「检索文档」中找到依据。

检索文档：
{docs}

AI 回复：
{reply}

评分规则（0-5分）：
- 5分：回复中所有事实陈述都能在检索文档中找到明确依据
- 3分：大部分有依据，有1-2处轻微推断但合理
- 1分：多处编造或与检索文档矛盾
- 0分：完全编造

只输出一个数字（0-5）。"""

RELEVANCE_PROMPT = """请判断以下 AI 回复是否真正回答了客户的问题。

客户问题：{question}

AI 回复：
{reply}

评分规则（0-5分）：
- 5分：直接、完整地回答了问题
- 3分：部分回答了，但有遗漏或跑题
- 1分：基本没回答，说了很多无关的话
- 0分：完全答非所问

只输出一个数字（0-5）。"""


def judge_with_llm(client: OpenAI, prompt: str) -> int:
    """用 LLM 打分，返回 0-5"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        # 提取数字
        import re
        nums = re.findall(r"\d+", text)
        if nums:
            score = int(nums[0])
            return max(0, min(5, score))
        return -1
    except Exception as e:
        print(f"   ⚠️ Judge 调用失败: {e}")
        return -1


# ============================================================
# 评估主逻辑
# ============================================================
async def run_evaluation(verbose: bool = False, use_llm: bool = True):
    """运行全链路评估"""

    # 初始化 rag
    import rag
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    rag.set_llm_client(client)
    rag.initialize()

    # 导入生成函数
    from app import _build_user_message, _call_llm
    from models import ReplyRequest
    from prompts import SYSTEM_PROMPT_REPLY

    results = []
    recall_scores = []
    faithfulness_scores = []
    relevance_scores = []
    negative_correct = 0
    negative_total = 0

    print(f"\n{'='*60}")
    print(f"🦪 珍珠 AI 助手 — RAG 全链路评估")
    print(f"{'='*60}")
    print(f"测试用例: {len(TEST_CASES)} 条")
    print(f"LLM Judge: {'启用' if use_llm else '仅检索'}\n")

    for tc in TEST_CASES:
        print(f"[{tc['id']}] {tc['question']}")

        # ── 1. 检索 ──
        docs, confidence = await rag.hybrid_search(tc["question"])
        all_retrieved_text = " ".join(docs)

        # Context Recall：期望关键词有多少出现在检索结果里
        if tc["expected_keywords"]:
            hits = sum(1 for kw in tc["expected_keywords"] if kw in all_retrieved_text)
            recall = hits / len(tc["expected_keywords"])
            recall_scores.append(recall)
            print(f"   检索 Recall: {hits}/{len(tc['expected_keywords'])} = {recall:.0%}")
        else:
            recall = None

        # ── 2. 负样本检测 ──
        if tc.get("is_negative"):
            negative_total += 1
            # 负样本：置信度应低于阈值，或生成内容应包含"不确定/查一下"
            is_low_conf = confidence < rag.SIMILARITY_FLOOR
            if is_low_conf:
                negative_correct += 1
                print(f"   负样本 ✅ 正确短路 (sim={confidence:.3f})")
            else:
                # 即使没短路，也检查生成内容
                req = ReplyRequest(question=tc["question"])
                user_msg = _build_user_message(req, docs)
                reply = _call_llm(SYSTEM_PROMPT_REPLY, user_msg)
                rejected = any(w in reply for w in ["不确定", "查一下", "不太确定", "帮你问"])
                if rejected:
                    negative_correct += 1
                    print(f"   负样本 ✅ 正确拒绝 (sim={confidence:.3f}, 生成内容含拒绝词)")
                else:
                    print(f"   负样本 ❌ 未正确拒绝 (sim={confidence:.3f})")
                    if verbose:
                        print(f"   回复: {reply[:120]}")
            results.append({"id": tc["id"], "recall": None, "faithfulness": None,
                            "relevance": None, "negative_correct": is_low_conf})
            continue

        # ── 3. 生成 + LLM 评分 ──
        if use_llm and docs:
            req = ReplyRequest(question=tc["question"])
            user_msg = _build_user_message(req, docs)
            reply = _call_llm(SYSTEM_PROMPT_REPLY, user_msg)

            # Faithfulness
            docs_text = "\n---\n".join(docs)
            faith_prompt = FAITHFULNESS_PROMPT.format(docs=docs_text[:2000], reply=reply)
            faith_score = judge_with_llm(client, faith_prompt)
            if faith_score >= 0:
                faithfulness_scores.append(faith_score)
                print(f"   Faithfulness: {faith_score}/5")

            # Relevance
            rel_prompt = RELEVANCE_PROMPT.format(question=tc["question"], reply=reply)
            rel_score = judge_with_llm(client, rel_prompt)
            if rel_score >= 0:
                relevance_scores.append(rel_score)
                print(f"   Relevance: {rel_score}/5")

            if verbose and docs:
                print(f"   检索结果 (top-1): {docs[0][:100]}...")
                print(f"   AI回复: {reply[:150]}...")

        results.append({"id": tc["id"], "recall": recall,
                        "faithfulness": faith_score if use_llm and docs else None,
                        "relevance": rel_score if use_llm and docs else None})

    # ── 汇总报告 ──
    print(f"\n{'='*60}")
    print(f"📊 评估报告")
    print(f"{'='*60}")

    if recall_scores:
        avg_recall = sum(recall_scores) / len(recall_scores)
        print(f"Context Recall:     {avg_recall:.0%}  (平均, {len(recall_scores)} 条)")

    if faithfulness_scores:
        avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
        print(f"Faithfulness:       {avg_faith:.1f}/5  (平均, {len(faithfulness_scores)} 条)")

    if relevance_scores:
        avg_rel = sum(relevance_scores) / len(relevance_scores)
        print(f"Answer Relevance:   {avg_rel:.1f}/5  (平均, {len(relevance_scores)} 条)")

    if negative_total > 0:
        neg_rate = negative_correct / negative_total
        print(f"Negative Rejection: {neg_rate:.0%}  ({negative_correct}/{negative_total})")

    # 综合评分
    scores_for_composite = []
    if recall_scores:
        scores_for_composite.append(sum(recall_scores) / len(recall_scores))
    if faithfulness_scores:
        scores_for_composite.append(sum(faithfulness_scores) / len(faithfulness_scores) / 5)
    if relevance_scores:
        scores_for_composite.append(sum(relevance_scores) / len(relevance_scores) / 5)
    if negative_total > 0:
        scores_for_composite.append(negative_correct / negative_total)

    if scores_for_composite:
        composite = sum(scores_for_composite) / len(scores_for_composite)
        print(f"\n{'─'*40}")
        print(f"🏆 综合评分: {composite:.0%}")
        if composite >= 0.8:
            print("   评级: A — 生产可用")
        elif composite >= 0.6:
            print("   评级: B — 基本可用，有改进空间")
        elif composite >= 0.4:
            print("   评级: C — 需要优化")
        else:
            print("   评级: D — 检索或知识库有严重问题")

    print(f"\n💡 面试话术:")
    if recall_scores:
        print(f"   '检索召回率 {avg_recall:.0%}，Faithfulness {avg_faith:.1f}/5，Negative Rejection {neg_rate:.0%}'")
    print(f"   '每个指标我都知道怎么测、怎么改进'")
    print()

    return results


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="珍珠 AI 助手 — RAG 评估")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印每条详情")
    parser.add_argument("--no-llm", action="store_true", help="只测检索，跳过 LLM 评分")
    args = parser.parse_args()

    asyncio.run(run_evaluation(verbose=args.verbose, use_llm=not args.no_llm))
