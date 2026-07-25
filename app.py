"""
app.py — 珍珠 AI 助手入口（路由 + 启动）
"""
import os
import json
import re
import time
import queue
import asyncio
import logging
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

from openai import OpenAI

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse

# ---- 拆分后的模块 ----
from models import ReplyRequest, ContentRequest, FeedbackRequest
from prompts import (
    SYSTEM_PROMPT_REPLY, SYSTEM_PROMPT_REPLY_SISTER, SYSTEM_PROMPT_CONTENT,
    build_customer_profile_prompt,
)
from feedback import log_feedback, get_adoption_stats
import rag
import inventory
from cache import cache
from db import db

# ---- 应用日志 ----
app_logger = logging.getLogger("pearl.app")
app_logger.setLevel(logging.DEBUG)
if not app_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[APP] %(message)s"))
    app_logger.addHandler(_h)
    app_logger.propagate = False


# ============================================================
# Agentic RAG 指标追踪
# ============================================================
class AgenticMetrics:
    """Agentic RAG 链路指标（线程安全）"""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_replies = 0
        self.short_circuits = 0
        self.self_check_triggered = 0       # 自检返回"有"
        self.refinement_searches = 0         # 实际发起了补充检索
        self.refinement_found_new = 0        # 补充检索找到了新文档
        self.hallucinated_attributions = 0   # 归因引用了不存在的条目

    def record_reply(self, *, short_circuit: bool = False,
                     self_check: bool = False, refinement: bool = False,
                     refinement_found: bool = False, attribution_hallucination: bool = False):
        with self._lock:
            self.total_replies += 1
            if short_circuit:
                self.short_circuits += 1
            if self_check:
                self.self_check_triggered += 1
            if refinement:
                self.refinement_searches += 1
            if refinement_found:
                self.refinement_found_new += 1
            if attribution_hallucination:
                self.hallucinated_attributions += 1

    def snapshot(self) -> dict:
        with self._lock:
            n = self.total_replies or 1
            return {
                "total_replies": self.total_replies,
                "short_circuit_rate": round(self.short_circuits / n, 3),
                "self_check_trigger_rate": round(self.self_check_triggered / n, 3),
                "refinement_rate": round(self.refinement_searches / max(self.self_check_triggered, 1), 3),
                "refinement_hit_rate": round(self.refinement_found_new / max(self.refinement_searches, 1), 3),
                "attribution_hallucination_rate": round(self.hallucinated_attributions / n, 3),
            }


_agentic_metrics = AgenticMetrics()


# ---- Agent Trace 日志 — ReAct 决策链结构化记录 ----
import uuid
import time as _time_module

_TRACE_LOG_FILE = Path(__file__).parent / "agent_trace.jsonl"
_trace_log_lock = threading.Lock()


def _save_trace(trace: dict):
    """持久化一条 Agent Trace 到 JSONL"""
    try:
        trace["ts"] = datetime.now(timezone.utc).isoformat()
        with _trace_log_lock:
            with open(_TRACE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---- MCP Client — 连接 MCP 库存服务 ----
_mcp_session = None
_mcp_stdio_ctx = None
_mcp_session_ctx = None


async def _init_mcp():
    """启动 MCP 库存 Server 并建立 Client 连接。
    通过 stdio 子进程通信，Agent 通过标准 MCP 协议发现和调用库存工具。
    """
    global _mcp_session, _mcp_stdio_ctx, _mcp_session_ctx
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession

    server_script = str(Path(__file__).parent / "mcp_inventory_server.py")
    server_params = StdioServerParameters(
        command="python",
        args=[server_script],
    )

    # 手动管理 context manager 生命周期（避免 AsyncExitStack 卡住）
    _mcp_stdio_ctx = stdio_client(server_params)
    read, write = await _mcp_stdio_ctx.__aenter__()
    _mcp_session_ctx = ClientSession(read, write)
    _mcp_session = await _mcp_session_ctx.__aenter__()
    await _mcp_session.initialize()
    print("[MCP] 库存服务连接成功 — Agent 可通过 MCP 协议发现和调用库存工具")


async def _shutdown_mcp():
    """关闭 MCP 连接和子进程"""
    global _mcp_session, _mcp_stdio_ctx, _mcp_session_ctx
    try:
        if _mcp_session_ctx:
            await _mcp_session_ctx.__aexit__(None, None, None)
    except Exception:
        pass
    try:
        if _mcp_stdio_ctx:
            await _mcp_stdio_ctx.__aexit__(None, None, None)
    except Exception:
        pass
    _mcp_session = None
    _mcp_stdio_ctx = None
    _mcp_session_ctx = None


# ============================================================
# Lifespan — 启动时加载模型，关闭时清理
# ============================================================
@asynccontextmanager
async def lifespan(application: FastAPI):
    """FastAPI 生命周期：启动加载模型，关闭清理资源"""
    print("\n🦪  珍珠 AI 助手 — 启动中...")
    try:
        # 注入 LLM 客户端
        _llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        rag.set_llm_client(_llm_client)
        # 加载模型和知识库（耗时操作，在 lifespan 里做不会阻塞 import）
        rag.initialize()
        # 启动 MCP 库存服务
        await _init_mcp()
        print("   服务就绪: http://localhost:8000\n")
    except Exception as e:
        print(f"   ❌ 启动失败: {e}")
        print("   请检查: 1) .env 里 DEEPSEEK_API_KEY 是否填写  2) chroma_db/ 是否存在（先跑 python build_knowledge.py）")
    yield
    # shutdown — 清理 MCP 连接
    await _shutdown_mcp()


app = FastAPI(title="珍珠 AI 助手", version="0.3.0", lifespan=lifespan)

# 方便其他模块引用（rag.rerank_with_deepseek 需要）
_llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
rag.set_llm_client(_llm_client)


# ============================================================
# 对话记忆（db.py — PostgreSQL 优先，JSON 文件自动兜底）
# ============================================================
MEMORY_TOP_K = 5

# 语义记忆用 jieba 做轻量关键词相关性
try:
    import jieba as _jieba
except ImportError:
    _jieba = None


def _get_memory(session_id: str) -> list:
    """读取原始历史（按时间顺序）"""
    return db.get_conversations(session_id)


def _get_relevant_memory(session_id: str, current_question: str) -> list:
    """语义记忆：从历史中找与当前问题最相关的轮次"""
    history = db.get_conversations(session_id)
    if not history:
        return []

    if len(history) <= MEMORY_TOP_K:
        return list(history)

    if _jieba:
        current_tokens = set(_jieba.cut(current_question))
    else:
        current_tokens = set(current_question)

    scored = []
    for i, h in enumerate(history):
        if _jieba:
            hist_tokens = set(_jieba.cut(h.get("question", "")))
        else:
            hist_tokens = set(h.get("question", ""))
        overlap = len(current_tokens & hist_tokens)
        recency = (i + 1) / len(history)
        score = overlap + recency * 1.5
        scored.append((score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:MEMORY_TOP_K]]


def _append_memory(session_id: str, question: str, reply: str):
    db.save_conversation(session_id, question, reply)


# ============================================================
# 归因验证 — 检查 LLM 引用的是不是真的查到了
# ============================================================
_ATTRIBUTION_PATTERN = re.compile(r"📎参考：(.+?)(?:\n|$)")

# 不确定时的兜底回复模板（按 user 区分）
_FALLBACK_REPLIES = {
    "mom": "这个问题我也不太确定，我帮你问问老板娘哈～等会儿回你 ❤️",
    "sister": "这个问题我不太确定，我帮你确认一下再回你。",
}


def _verify_attribution(reply: str, docs: list[str]) -> dict:
    """解析回复里的归因行，验证引用的条目号是否在检索到的文档范围内。

    返回 {"cited": [1, 3], "valid": [1, 3], "hallucinated": [], "doc_count": 5}
    hallucinated 表示 LLM 声称引用了不存在的条目号 — 这就是幻觉信号。
    """
    result = {"cited": [], "valid": [], "hallucinated": [], "doc_count": len(docs)}
    match = _ATTRIBUTION_PATTERN.search(reply)
    if not match:
        return result

    raw = match.group(1).strip()
    # "条目1、条目3" 或 "珍珠通用知识" → 前者才验证
    numbers = re.findall(r"\d+", raw)
    if not numbers:
        return result

    result["cited"] = [int(n) for n in numbers]
    for n in result["cited"]:
        if 1 <= n <= len(docs):
            result["valid"].append(n)
        else:
            result["hallucinated"].append(n)

    if result["hallucinated"]:
        app_logger.warning(
            "⚠️ 归因幻觉！LLM 引用了条目 %s，但实际只有 %d 条文档",
            result["hallucinated"], len(docs),
        )

    return result


# ============================================================
# Agentic RAG：检索 → 生成 → 自检 → 再检索 → 修正
# ============================================================
def _call_llm(system: str, user: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """调用 DeepSeek，返回纯文本"""
    resp = _llm_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _call_llm_chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """调用 DeepSeek（多轮对话），返回纯文本"""
    try:
        resp = _llm_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        finish = resp.choices[0].finish_reason
        content = resp.choices[0].message.content
        app_logger.info("📡 DeepSeek 返回: finish=%s len=%d", finish, len(content or ""))
        if not content:
            app_logger.warning("⚠️ DeepSeek 返回空内容, finish_reason=%s", finish)
            return ""
        return content.strip()
    except Exception as e:
        app_logger.error("❌ DeepSeek 调用失败: %s", e)
        return ""


def _build_user_message(req: ReplyRequest, docs: list[str]) -> str:
    """构建发给 LLM 的用户消息：客户画像 + 知识库文档 + 客户问题。
    evaluate.py 依赖此函数做全链路评估。
    """
    user_msg = f"客户问题：{req.question}"

    profile = build_customer_profile_prompt(
        req.trait, req.knowledge_level, req.budget_range, req.usage, req.quality,
    )
    if profile:
        user_msg = profile + "\n\n" + user_msg

    if docs:
        kb_text = "\n\n".join(
            f"[知识条目{i + 1}]\n{d}" for i, d in enumerate(docs)
        )
        user_msg = f"【知识库参考】\n{kb_text}\n\n{user_msg}"

    return user_msg


# ============================================================
# ReAct Agent — 工具调度层
# LLM 自主决定调哪些工具、调几次、什么顺序。
#
# 面试话术：
# "Agent 部分没用 LangGraph，自己写了 ReAct 循环。
#  工具做了注册制——TOOLS 字典统一管理，加新工具只需加一行配置，
#  不改 _react_system_prompt 和 _execute_tool。
#  生成后加了一轮自检：编造检查、预算检查、款式检查，
#  相当于 Agent 自己审核自己。纠错比人快。"
# ============================================================

# ============================================================
# 工具注册表 — 加新工具只需在这里加一行
# Agent 循环逻辑与工具定义解耦，不改 _react_system_prompt / _execute_tool
# ============================================================

async def _handle_search_knowledge(args: dict) -> tuple[str, list[str]]:
    """搜索珍珠知识库"""
    query = args.get("query", "")
    docs, conf = await rag.hybrid_search(query)
    if conf < rag.SIMILARITY_FLOOR:
        return "⚠️ 知识库中暂无高度匹配的内容。请在回复中诚实告知客户，不要编造。", []
    return "\n\n".join(
        f"[知识条目{i + 1}]\n{d}" for i, d in enumerate(docs)
    ), docs


async def _handle_check_inventory(args: dict) -> tuple[str, list[str]]:
    """查询库存 — 通过 MCP 协议调用库存服务（而非直接 import inventory）"""
    kw = args.get("keyword", "")
    # 通过 MCP 协议调用库存工具
    if _mcp_session:
        try:
            result = await _mcp_session.call_tool("check_inventory", {"query": kw})
            items_text = result.content[0].text if result.content else ""
            if not items_text or "暂无匹配" in items_text:
                return (
                    f"⚠️ 库存中没有找到「{kw}」。请用更短更宽泛的关键词（只搜品种名'Akoya'），"
                    "调用 check_inventory 再查一次。查不到就诚实说'这个姐帮你找找看'。"
                ), []
            text = f"🔍 搜「{kw}」→ MCP 库存服务返回：\n\n{items_text}"
            text += (
                "\n⚠️ 重要：客户没主动问价格就不要报价格！只介绍品种、品质、适合谁。"
                "\n如果客户问了多少钱，再用上面的实际单价来报。"
                "\n只推荐珠子（品种+尺码），不要推荐成品款式（全珠链/吊坠/耳钉等）。"
            )
            return text, []
        except Exception as e:
            app_logger.warning("MCP 库存调用失败，降级到直连: %s", e)
    # 降级：MCP 不可用时直连 inventory
    items = inventory.search(kw)
    if not items:
        return (
            f"⚠️ 库存中没有找到「{kw}」。不要自己编产品名和价格！"
            "请用更短更宽泛的关键词（比如只搜品种名'Akoya'，去掉尺码和款式词），"
            "调用 check_inventory 再查一次。如果两次都查不到，就诚实告诉客户'这个姐帮你找找看'。"
        ), []
    text = inventory.format_inventory_for_prompt(items)
    text = f"🔍 搜「{kw}」→ 匹配到以下库存（只有这些，没有其他）：\n\n{text}"
    text += (
        "\n⚠️ 重要：客户没主动问价格就不要报价格！只介绍品种、品质、适合谁。"
        "\n如果客户问了多少钱，再用上面的实际单价来报。"
        "\n只推荐珠子（品种+尺码），不要推荐成品款式（全珠链/吊坠/耳钉等）。"
        "\n⚠️ 上面列出的就是所有有货的库存。客户问的精确尺寸没匹配到时，推荐最接近的尺寸，不要说'没货'——你看到的列表就是有货的。"
    )
    return text, []


TOOLS: dict[str, dict] = {
    "search_knowledge": {
        "description": "搜索珍珠知识库，获取专业知识",
        "parameters": {"query": "搜索关键词"},
        "handler": _handle_search_knowledge,
    },
    "check_inventory": {
        "description": "查询库存，获取珠子品种、规格、价格、数量",
        "parameters": {"keyword": "珠子品种或关键词"},
        "handler": _handle_check_inventory,
    },
}


def _react_system_prompt(user_type: str) -> str:
    """构建包含工具清单的 System Prompt（从 TOOLS 注册表动态生成）"""
    base = SYSTEM_PROMPT_REPLY if user_type == "mom" else SYSTEM_PROMPT_REPLY_SISTER
    tools_desc = "\n".join(
        f"  • {name} — {info['description']}  参数：{json.dumps(info['parameters'], ensure_ascii=False)}"
        for name, info in TOOLS.items()
    )
    # 动态生成 JSON 示例
    examples = "\n".join(
        f'{{{{"tool":"{name}","args":{json.dumps(info["parameters"], ensure_ascii=False)}}}}}'
        for name, info in TOOLS.items()
    )
    return f"""{base}

【可用工具 — 你可自主决定是否调用、调哪个、调几次】
{tools_desc}

【工作方式】
- 下面消息中的知识库/库存是系统初步检索的结果，作为参考起点
- 如果觉得信息不够或角度不对，调用 search_knowledge 换关键词再搜
- 客户有购买意向 → 调用 check_inventory 查库存
- 信息足够后直接输出纯文本回复
- 调工具时只输出一行 JSON，如：
{examples}

【生成后自检 — 最高优先级】
回复客户前，在脑中逐条检查：
1. 产品名、价格是否全部来自库存数据？→ 不是则删掉
2. 是否超出客户预算？→ 超了则换便宜的
3. 推荐的是珠子品种（如'Akoya 7-8mm'）还是成品款式（全珠链/吊坠/耳钉）？→ 只推珠子
4. 有没有编造知识库没有的内容？→ 有则删掉
有问题自己修正后再输出，不要在回复里写检查过程。

【结尾铁律】
1. 客户说想买什么就推什么，不替客户拒绝
2. 预算范围内推最好的，不推远低于预算的便宜货
3. 客户没问价格不报价格
4. 推荐完直接停，不反问客户问题
5. 不编造、不虚构——不知道就说"姐帮你找找"
"""



def _parse_action(text: str) -> dict | None:
    """从 LLM 回复中提取 JSON 工具调用。
    支持两种情况：整段文本就是 JSON，或者 JSON 嵌在文本中间。
    返回 None 表示 LLM 没有发起工具调用（即最终回复）。
    """
    text = text.strip()
    print(f"[DEBUG _parse_action] 收到 (前200字): {text[:200]}", flush=True)
    # 情况1：整段就是 JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            app_logger.info("✅ 解析为工具调用: %s", obj.get("tool"))
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    # 情况2：文本中嵌了 JSON（找 {"tool" 开头的最外层大括号）
    for m in re.finditer(r'\{\s*"tool"\s*:', text):
        depth, start = 0, m.start()
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict) and "tool" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


async def _execute_tool(name: str, args: dict) -> tuple[str, list[str]]:
    """执行工具 — 从 TOOLS 注册表查找 handler，不硬编码 if-else 路由。
    返回 (LLM 可读的格式化结果, 知识库文档列表用于归因验证)。
    """
    tool = TOOLS.get(name)
    if tool is None:
        available = ", ".join(TOOLS.keys())
        return f"未知工具「{name}」。可用：{available}", []
    return await tool["handler"](args)


async def _agentic_reply(req: ReplyRequest) -> tuple[str, list[str]]:
    """ReAct Agent：LLM 自主决策工具调用 → 执行 → 观察 → 再决策 → 最终回复

    替换了原来硬编码的"先查知识库再查库存"流程。
    LLM 可以根据问题类型只查知识库（知识类问题），或者先查知识库再查库存（推荐类问题），
    甚至可以换关键词多次检索——所有决策由 LLM 在 ReAct 循环里完成。
    """
    trace_id = uuid.uuid4().hex[:12]
    trace_steps: list[dict] = []
    t_start = _time_module.perf_counter()

    # ---- 初始消息：客户画像 + 语义记忆 ----
    user_msg = f"客户问题：{req.question}"

    profile = build_customer_profile_prompt(
        req.trait, req.knowledge_level, req.budget_range, req.usage, req.quality,
    )
    if profile:
        user_msg = profile + "\n\n" + user_msg

    all_docs: list[str] = []
    had_low_conf = False
    tool_used = False

    # ---- 强制预检索：知识库 + 库存，不等 LLM 决定 ----
    # 下拉框的每个维度都喂进搜索，不只是文本框
    search_terms = [req.question]
    for val in [req.usage, req.quality, req.knowledge_level, req.trait]:
        if val.strip():
            search_terms.append(val.strip())
    full_query = " ".join(search_terms)
    kb_docs, kb_conf = await rag.hybrid_search(full_query)
    if kb_conf >= rag.SIMILARITY_FLOOR:
        all_docs.extend(kb_docs)
        kb_text = "\n\n".join(
            f"[知识条目{i + 1}]\n{d}" for i, d in enumerate(kb_docs)
        )
        retrieval_msg = f"【知识库初步检索 — 作为参考，你仍可自行搜索】\n{kb_text}\n\n{user_msg}"
    else:
        had_low_conf = True
        retrieval_msg = user_msg

    # 预查库存 — 知识库关键词 → 精准搜库存
    kb_keywords = _extract_kb_keywords(kb_docs) if kb_docs else []
    inv_items, budget_note = _budget_aware_inventory_search(req.question, kb_keywords)
    if inv_items:
        inv_text = inventory.format_inventory_for_prompt(inv_items)
        inv_text += "\n⚠️ 上面就是当前所有有货的库存。只推荐这里面的珠子，不要编造。"
        inv_text += budget_note
        retrieval_msg += f"\n\n【当前库存（自动查询）】\n{inv_text}"
    else:
        app_logger.info("📦 预查库存 — 无匹配结果，query=%s", req.question[:60])

    messages: list[dict] = [
        {"role": "system", "content": _react_system_prompt(req.user)},
        {"role": "user", "content": retrieval_msg},
    ]

    # ---- ReAct 循环：LLM 决定是否还要查库存 ----
    for turn in range(3):
        resp = _call_llm_chat(messages)
        action = _parse_action(resp)

        if action is None:
            # 记录最终轮 — LLM 决定直接回复，不调工具
            trace_steps.append({
                "round": turn + 1, "decision": "reply",
                "response_preview": resp[:120],
            })
            # ---- 自检轮：Agent 审核自己的回复 ----
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content": (
                "【自检】逐条确认：1) 产品名和价格是否全部来自上面的库存数据？"
                "2) 有没有超出客户预算？"
                "3) 推荐的是珠子品种还是成品款式？只推珠子。"
                "4) 有没有编造知识库没有的内容？"
                "如有问题，输出修正后的回复；如无问题，直接输出原回复。"
                "不要输出检查过程，只输出最终回复。"
            )})
            checked = _call_llm_chat(messages, temperature=0.3)

            # 防 JSON 泄漏：自检结果不能是工具调用
            if _parse_action(checked) is not None or checked.strip().startswith("{"):
                checked = resp

            verif = _verify_attribution(checked, all_docs)
            _agentic_metrics.record_reply(
                short_circuit=had_low_conf,
                self_check=tool_used,
                attribution_hallucination=bool(verif["hallucinated"]),
            )
            # 持久化 Agent Trace
            _save_trace({
                "trace_id": trace_id,
                "question": req.question,
                "pre_retrieval_docs": len(kb_docs) if kb_docs else 0,
                "pre_inventory_items": len(inv_items),
                "steps": trace_steps,
                "self_check": {"triggered": tool_used, "hallucinated_refs": verif["hallucinated"]},
                "total_rounds": turn + 1,
                "latency_ms": round((_time_module.perf_counter() - t_start) * 1000, 1),
            })
            return _strip_meta_lines(checked, all_docs), all_docs

        # 执行 LLM 请求的工具（主要是 check_inventory）
        tool_used = True
        tool_name = action.get("tool", "")
        tool_args = action.get("args", {})
        obs, docs = await _execute_tool(tool_name, tool_args)
        all_docs.extend(docs)

        # 记录 ReAct 工具调用 trace
        trace_steps.append({
            "round": turn + 1,
            "decision": "tool_call",
            "tool": tool_name,
            "args": tool_args,
            "observation_len": len(obs),
        })

        if tool_name == "search_knowledge" and "暂无匹配" in obs:
            had_low_conf = True

        app_logger.info(
            "🔧 ReAct 第%d轮：%s(%s)",
            turn + 1, tool_name, json.dumps(tool_args, ensure_ascii=False),
        )

        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "user", "content": (
            f"「{tool_name}」返回结果：\n{obs}\n\n"
            "（如果还需要查其他工具，继续输出 JSON；如果信息足够回复客户，直接输出纯文本回复）"
        )})

    # ---- 兜底：3 轮后强制回复 ----
    messages.append({"role": "user", "content": "请综合以上所有信息，用口语化的短句直接回复客户。"})
    final = _call_llm_chat(messages)
    if _parse_action(final) is not None or final.strip().startswith("{"):
        messages.append({"role": "user", "content": "不要输出 JSON！用纯文本口语直接回复客户。"})
        final = _call_llm_chat(messages)
    _save_trace({
        "trace_id": trace_id,
        "question": req.question,
        "pre_retrieval_docs": len(kb_docs) if kb_docs else 0,
        "pre_inventory_items": len(inv_items),
        "steps": trace_steps,
        "self_check": {"triggered": False, "hallucinated_refs": []},
        "total_rounds": 3,
        "force_reply": True,
        "latency_ms": round((_time_module.perf_counter() - t_start) * 1000, 1),
    })
    if final.strip().startswith("{"):
        return "抱歉，我这边信息有点多，让我整理一下再回复您～", all_docs
    return _strip_meta_lines(final, all_docs), all_docs


# ============================================================
# 统一错误包装
# ============================================================
def _error_reply(message: str, status_code: int = 500) -> JSONResponse:
    """返回中文错误信息，避免裸 traceback 泄露给前端"""
    return JSONResponse(
        status_code=status_code,
        content={"error": message},
    )


# ============================================================
# 后处理：过滤 LLM 回复中的元数据行
# prompt 规则不一定被遵守，代码过滤才可靠
# ============================================================
_REF_PATTERN = re.compile(r"📎参考：")
_PUSHY_PATTERNS = [
    re.compile(r".*姐给你.*?最实在的价.*"),
    re.compile(r".*姐给你老客户.*?价.*"),
    re.compile(r".*[一二三四五六七八九十]颗\d+.*?姐给你.*?(?:价|安排).*"),
    re.compile(r".*要的话直接说.*?(?:发走|安排|发货).*"),
    re.compile(r".*要几颗.*"),
    re.compile(r".*立马安排发[走货].*"),
    re.compile(r".*姐直接安排最好的给你.*"),
    re.compile(r".*姐给你安排.*"),
    re.compile(r".*姐.*?最实在的.*"),
    re.compile(r".*肯定喜欢.*"),
    re.compile(r".*一定喜欢.*"),
    re.compile(r".*绝对.*?喜欢.*"),
]


def _strip_meta_lines(text: str, docs: list[str] | None = None) -> str:
    """删除回复中的推销话术。📎参考保留发给用户（复制时手动删）。"""
    lines = text.split("\n")
    # 过滤前先解析 📎参考 行，打印真正的知识库条目（调试用）
    for l in lines:
        m = _REF_PATTERN.search(l.strip())
        if m and docs:
            # 从 "📎参考：条目1、条目3" 提取条目号
            nums = [int(n) for n in re.findall(r'\d+', l)]
            for n in nums:
                if 1 <= n <= len(docs):
                    title = docs[n - 1].split('\n')[0].strip()
                    # 知识库格式 "问：..." → 提取问句
                    q_match = re.search(r'问：(.+?)(?:答|$)', docs[n - 1])
                    if q_match:
                        title = q_match.group(1).strip()[:60]
                    app_logger.info("📎 引用条目%d: %s", n, title)
        elif m:
            app_logger.info("📎 %s", l.strip())
    # 过滤掉匹配推销模式的行（📎参考 保留，发给客户前手动删掉即可）
    lines = [l for l in lines
             if not any(p.search(l.strip()) for p in _PUSHY_PATTERNS)]
    # 去掉末尾空行
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# ============================================================
# 预算提取 + 库存过滤 — 代码层硬约束，不靠 LLM 自觉
# ============================================================

_BUDGET_PATTERNS = [
    re.compile(r'预算\s*(\d+)\s*[-–—到至]\s*(\d+)'),        # 预算2000-3000
    re.compile(r'预算\s*不?超过?\s*(\d+)'),                   # 预算不超过2000
    re.compile(r'预算\s*(\d+)\s*以[内下]'),                   # 预算2000以内
    re.compile(r'预算\s*(\d+)\s*以[上外]'),                   # 预算2000以上
    re.compile(r'预算\s*(\d+)\s*左?右'),                     # 预算2000左右
    re.compile(r'预算\s*(\d+)'),                              # 预算2000（兜底）
    re.compile(r'(\d+)\s*[-–—到至]\s*(\d+)\s*(?:块|元)?.*?(?:预算|想买|价位|价格)'),  # 2000-3000预算
    re.compile(r'(?:^|\s)(\d{3,5})(?:块|元|块钱)?(?:\s|$|，|。|！|？)'),  # 纯数字 3000 / 3000块
]


def _extract_budget(question: str) -> tuple[float | None, float | None]:
    """从客户问题中提取预算范围，返回 (min, max)。

    例：
    "预算2000-3000" → (2000, 3000)
    "预算不超过1000" → (None, 1000)
    "预算5000以上" → (5000, None)
    "预算2000左右" → (1400, 2600)  # ±30%
    "预算2000" → (2000, 2000)
    """
    for pat in _BUDGET_PATTERNS:
        m = pat.search(question)
        if m is None:
            continue
        groups = m.groups()
        if len(groups) == 2 and groups[1] is not None:
            return float(groups[0]), float(groups[1])

        val = float(groups[0])
        # 判断预算修饰词
        if re.search(r'(?:以[内下]|不超过|不超)', question):
            return None, val
        if re.search(r'(?:以[上外])', question):
            return val, None
        if re.search(r'(?:左右|大约|大概|差不多)', question):
            return val * 0.7, val * 1.3
        # 纯数字（"预算3000"）→ 预算就是3000左右，下限别太低
        return val * 0.5, val * 1.1

    return None, None


def _filter_by_budget(items: list[dict], min_b: float | None, max_b: float | None) -> list[dict]:
    """按预算过滤库存。只保留价格在预算范围内的珠子。

    - 有下限时：价格 >= 下限（_extract_budget 已做弹性，这里不再打折）
    - 有上限时：价格 <= 上限*1.1（留 10% 弹性）
    """
    if min_b is None and max_b is None:
        return items

    filtered = []
    for item in items:
        price = item.get("price_per_piece", 0)
        if price <= 0:
            continue
        if min_b is not None and price < min_b:
            continue
        if max_b is not None and price > max_b * 1.1:
            continue
        filtered.append(item)
    return filtered


# 从知识库文档中提取珍珠品种关键词，用于精准搜库存
_PEARL_TYPES = re.compile(
    r'(澳白|南洋金珠|金珠|南洋白珠|大溪地|Akoya|akoya|真多麻|马贝|淡水珍珠|淡水|海水珍珠|海水)'
)

def _extract_kb_keywords(docs: list[str]) -> list[str]:
    """从知识库检索结果中提取珍珠品种，用作库存搜索关键词"""
    seen = set()
    keywords = []
    for doc in docs:
        for m in _PEARL_TYPES.findall(doc):
            # 统一：金珠→南洋金珠
            kw = "南洋金珠" if m in ("金珠", "南洋金珠") else m
            if kw not in seen:
                seen.add(kw)
                keywords.append(kw)
    return keywords


def _budget_aware_inventory_search(question: str, kb_keywords: list[str] | None = None) -> tuple[list[dict], str]:
    """预算感知的库存搜索：
    1. 关键词搜索库存（优先用知识库提取的品种关键词，没有则用原问题）
    2. 从问题中提取预算
    3. 按预算过滤
    4. 如果过滤后结果太少，用全量库存补充预算内产品
    5. 返回 (过滤后库存列表, 预算提示文本)
    """
    # 有知识库提取的品种关键词时，用它搜库存（比原问题更精准）
    search_query = question
    if kb_keywords:
        search_query = " ".join(kb_keywords)
        app_logger.info("📦 用知识库关键词搜库存: %s", search_query)
    keyword_items = inventory.search(search_query)
    min_b, max_b = _extract_budget(question)

    if min_b is None and max_b is None:
        # 关键词没匹配到，预算也没提取到 → 兜底返回全部库存
        if not keyword_items:
            all_items = [i for i in inventory.list_all() if i.get('stock', 0) > 0]
            app_logger.info("📦 预算/关键词均未匹配，兜底返回全量库存 %d 条", len(all_items))
            return all_items, ""
        return keyword_items, ""

    # 过滤关键词搜索结果
    filtered = _filter_by_budget(keyword_items, min_b, max_b)

    # 如果关键词匹配不够，从全量库存中补预算内的产品
    if len(filtered) < 3:
        all_items = inventory.list_all()
        all_in_budget = _filter_by_budget(all_items, min_b, max_b)
        existing_names = {item['name'] for item in filtered}
        for item in all_in_budget:
            if item['name'] not in existing_names and item.get('stock', 0) > 0:
                filtered.append(item)
                existing_names.add(item['name'])

    # 生成预算提示
    budget_note = ""
    if max_b and min_b:
        budget_note = f"\n💰 客户预算 {int(min_b)}-{int(max_b)} 元。只推荐这个价格范围内的珠子，不要推超出预算的。"
    elif max_b:
        budget_note = f"\n💰 客户预算不超过 {int(max_b)} 元。只推荐这个价格范围内的珠子，不要推超出预算的。"
    elif min_b:
        budget_note = f"\n💰 客户预算 {int(min_b)} 以上。只推荐这个价位及以上的珠子。"

    return filtered, budget_note


# ============================================================
# SSE 工具
# ============================================================
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_llm_stream(messages: list[dict], temperature: float,
                    max_tokens: int, out_q: queue.Queue):
    """在线程中跑 DeepSeek 流式调用，token 逐个放入队列。
    队列中放入 ("token", str) 或 ("done", None) 或 ("error", str)。"""
    try:
        resp = _llm_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                out_q.put(("token", delta))
        out_q.put(("done", None))
    except Exception as exc:
        out_q.put(("error", str(exc)))


async def _stream_tokens(messages: list[dict],
                         temperature: float = 0.7,
                         max_tokens: int = 2048) -> AsyncGenerator[str, None]:
    """异步迭代器：从 DeepSeek 流中逐 token 产出，不阻塞事件循环。"""
    q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_run_llm_stream,
        args=(messages, temperature, max_tokens, q),
        daemon=True,
    )
    thread.start()

    loop = asyncio.get_running_loop()
    while True:
        msg_type, payload = await loop.run_in_executor(None, q.get)
        if msg_type == "token":
            yield payload
        elif msg_type == "done":
            break
        elif msg_type == "error":
            app_logger.error("DeepSeek 流式调用失败: %s", payload)
            break
    thread.join(timeout=5)


@app.post("/api/reply")
async def api_reply(req: ReplyRequest):
    try:
        reply, docs = await _agentic_reply(req)
        return {"reply": reply, "sources": docs}
    except RuntimeError as e:
        return _error_reply(f"知识库还没准备好：{e}", 503)
    except Exception as e:
        print(f"[ERROR] /api/reply: {e}")
        return _error_reply("生成回复时出错了，请稍后重试")


@app.post("/api/reply/stream")
async def api_reply_stream(req: ReplyRequest):
    """ReAct 流式：初步检索 → Agent 自主决策工具调用 → 生成 → 自检 → 流式输出。"""


    async def event_stream():
        trace_id = uuid.uuid4().hex[:12]
        trace_steps: list[dict] = []
        t_start = _time_module.perf_counter()
        try:
            # ---- 构建初始消息（客户画像 + 语义记忆） ----
            user_msg = f"客户问题：{req.question}"

            profile = build_customer_profile_prompt(
                req.trait, req.knowledge_level, req.budget_range, req.usage, req.quality,
            )
            if profile:
                user_msg = profile + "\n\n" + user_msg

            all_docs: list[str] = []
            had_low_conf = False
            tool_used = False

            # ---- 强制预检索：知识库 + 库存，不等 LLM 决定 ----
            # 下拉框的每个维度都喂进搜索
            yield _sse({"status": "searching", "tool": "search_knowledge"})
            search_terms = [req.question]
            for val in [req.usage, req.quality, req.knowledge_level, req.trait]:
                if val.strip():
                    search_terms.append(val.strip())
            full_query = " ".join(search_terms)
            kb_docs, kb_conf = await rag.hybrid_search(full_query)
            if kb_conf >= rag.SIMILARITY_FLOOR:
                all_docs.extend(kb_docs)
                kb_text = "\n\n".join(
                    f"[知识条目{i + 1}]\n{d}" for i, d in enumerate(kb_docs)
                )
                retrieval_msg = f"【知识库初步检索 — 作为参考，你仍可自行搜索】\n{kb_text}\n\n{user_msg}"
            else:
                had_low_conf = True
                retrieval_msg = user_msg

            # 预查库存 — 知识库关键词 → 精准搜库存
            kb_keywords = _extract_kb_keywords(kb_docs) if kb_docs else []
            inv_items, budget_note = _budget_aware_inventory_search(req.question, kb_keywords)
            if inv_items:
                app_logger.info("📦 预查库存 — 匹配 %d 条: %s", len(inv_items),
                                ", ".join(i['name'] for i in inv_items[:5]))
                inv_text = inventory.format_inventory_for_prompt(inv_items)
                inv_text += "\n⚠️ 上面就是当前所有有货的库存。只推荐这里面的珠子，不要编造。"
                inv_text += budget_note
                retrieval_msg += f"\n\n【当前库存（自动查询）】\n{inv_text}"
            else:
                app_logger.info("📦 预查库存 — 无匹配结果，query=%s", req.question[:60])

            messages: list[dict] = [
                {"role": "system", "content": _react_system_prompt(req.user)},
                {"role": "user", "content": retrieval_msg},
            ]

            # ---- ReAct 循环：LLM 决定是否还要查库存 ----
            for turn in range(3):
                yield _sse({"status": "thinking"})
                resp = _call_llm_chat(messages)
                action = _parse_action(resp)

                if action is None:
                    trace_steps.append({
                        "round": turn + 1, "decision": "reply",
                        "response_preview": resp[:120],
                    })
                    # ---- 自检轮：Agent 审核自己的回复 ----
                    yield _sse({"status": "checking"})
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": (
                        "【自检】逐条确认：1) 产品名和价格是否全部来自上面的库存数据？"
                        "2) 有没有超出客户预算？"
                        "3) 推荐的是珠子品种还是成品款式？只推珠子。"
                        "4) 有没有编造知识库没有的内容？"
                        "如有问题，输出修正后的回复；如无问题，直接输出原回复。"
                        "不要输出检查过程，只输出最终回复。"
                    )})
                    checked = _call_llm_chat(messages, temperature=0.3)
                    if _parse_action(checked) is not None or checked.strip().startswith("{"):
                        checked = resp

                    # 流式输出自检后的回复
                    yield _sse({"status": "generating"})
                    final_reply = _strip_meta_lines(checked, all_docs)
                    if not final_reply.strip():
                        final_reply = "不好意思宝，刚刚信息有点多，姐重新帮你看一下哈 ❤️"
                    for i in range(0, len(final_reply), 3):
                        yield _sse({"token": final_reply[i:i+3]})
                        await asyncio.sleep(0.02)
                    verif = _verify_attribution(final_reply, all_docs)
                    _agentic_metrics.record_reply(
                        short_circuit=had_low_conf,
                        self_check=tool_used,
                        attribution_hallucination=bool(verif["hallucinated"]),
                    )
                    _save_trace({
                        "trace_id": trace_id,
                        "question": req.question,
                        "pre_retrieval_docs": len(kb_docs) if kb_docs else 0,
                        "pre_inventory_items": len(inv_items),
                        "steps": trace_steps,
                        "self_check": {"triggered": tool_used, "hallucinated_refs": verif["hallucinated"]},
                        "total_rounds": turn + 1,
                        "latency_ms": round((_time_module.perf_counter() - t_start) * 1000, 1),
                    })
                    yield _sse({"done": True, "full_reply": final_reply})
                    return

                # 执行 LLM 请求的工具
                tool_used = True
                tool_name = action.get("tool", "")
                tool_args = action.get("args", {})
                yield _sse({"status": "searching", "tool": tool_name})

                obs, docs = await _execute_tool(tool_name, tool_args)
                all_docs.extend(docs)

                trace_steps.append({
                    "round": turn + 1,
                    "decision": "tool_call",
                    "tool": tool_name,
                    "args": tool_args,
                    "observation_len": len(obs),
                })

                if tool_name == "search_knowledge" and "暂无匹配" in obs:
                    had_low_conf = True

                app_logger.info(
                    "🔧 ReAct stream 第%d轮：%s(%s)",
                    turn + 1, tool_name, json.dumps(tool_args, ensure_ascii=False),
                )

                messages.append({"role": "assistant", "content": resp})
                messages.append({"role": "user", "content": (
                    f"「{tool_name}」返回结果：\n{obs}\n\n"
                    "（还需要查工具就输出 JSON，信息够了直接回复客户）"
                )})

            # ---- 兜底：3 轮后强制流式回复 ----
            yield _sse({"status": "generating"})
            messages.append({"role": "user", "content": (
                "请综合以上所有信息，用口语化的短句直接回复客户。"
                "务必输出纯文本——不要输出 JSON、不要调用工具、不要用代码块。"
            )})
            final_reply = ""
            async for token in _stream_tokens(messages):
                final_reply += token
                yield _sse({"token": token})

            # 防 JSON 泄漏：如果 LLM 在兜底轮仍然返回了 JSON，再试一次
            if final_reply.strip().startswith("{"):
                app_logger.warning("兜底轮返回了 JSON，追加更强调的指令重试")
                messages.append({"role": "assistant", "content": final_reply})
                messages.append({"role": "user", "content": "不对——请用口语短句直接回复，不要 JSON！"})
                final_reply = ""
                async for token in _stream_tokens(messages):
                    final_reply += token
                    yield _sse({"token": token})

            # 兜底路径也过滤推销话术
            final_reply = _strip_meta_lines(final_reply, all_docs)
            _save_trace({
                "trace_id": trace_id,
                "question": req.question,
                "pre_retrieval_docs": len(kb_docs) if kb_docs else 0,
                "pre_inventory_items": len(inv_items),
                "steps": trace_steps,
                "self_check": {"triggered": False, "hallucinated_refs": []},
                "total_rounds": 3,
                "force_reply": True,
                "latency_ms": round((_time_module.perf_counter() - t_start) * 1000, 1),
            })
            yield _sse({"done": True, "full_reply": final_reply})

        except RuntimeError as e:
            yield _sse({"error": f"知识库还没准备好：{e}"})
        except Exception as e:
            app_logger.error("Stream 异常: %s", e)
            yield _sse({"error": "生成回复时出错了，请稍后重试"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/content")
async def api_content(req: ContentRequest):
    try:
        docs, confidence = await rag.hybrid_search(req.topic)
        inv_items = inventory.search(req.topic)

        # 置信度过低 → 知识库不够，诚实告知
        if confidence < rag.SIMILARITY_FLOOR:
            app_logger.info(
                "⛔ 内容生成短路 — topic='%s' sim=%.3f < floor=%.2f",
                req.topic[:60], confidence, rag.SIMILARITY_FLOOR,
            )
            return {
                "content": "目前知识库里关于这个话题的信息还不够，等老板娘补充了相关资料我再帮你写～",
                "sources": [],
            }

        knowledge_text = "\n\n".join(
            f"[知识库条目 {i+1}]\n{doc}" for i, doc in enumerate(docs)
        )
        user_msg = f"请写一篇关于「{req.topic}」的小红书笔记。\n\n以下是珍珠知识库中相关内容，请以此为素材：\n\n{knowledge_text}"

        resp = _llm_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CONTENT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content.strip()
        return {"content": content, "sources": docs}
    except RuntimeError as e:
        return _error_reply(f"知识库还没准备好：{e}", 503)
    except Exception as e:
        print(f"[ERROR] /api/content: {e}")
        return _error_reply("生成内容时出错了，请稍后重试")


@app.post("/api/feedback")
async def api_feedback(req: FeedbackRequest):
    try:
        stats = log_feedback(req.action, req.question, req.reply)
        return {"ok": True, "stats": stats}
    except Exception as e:
        print(f"[ERROR] /api/feedback: {e}")
        return _error_reply("记录反馈失败")


@app.get("/api/stats")
async def api_stats():
    try:
        return get_adoption_stats()
    except Exception as e:
        print(f"[ERROR] /api/stats: {e}")
        return _error_reply("获取统计数据失败")


@app.get("/api/metrics")
async def api_metrics():
    """全链路指标：检索 + Agentic RAG + 采纳率"""
    try:
        retrieval = rag.get_retrieval_metrics()
        agentic = _agentic_metrics.snapshot()
        adoption = get_adoption_stats()
        return {
            "retrieval": retrieval,
            "agentic_rag": agentic,
            "adoption": adoption,
        }
    except Exception as e:
        print(f"[ERROR] /api/metrics: {e}")
        return _error_reply("获取指标失败")


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html 还没创建</h1>", status_code=404)


# ---- 库存管理 API ----

@app.get("/api/inventory")
async def api_inventory_list():
    """查看全部库存"""
    try:
        items = inventory.list_all()
        return {"count": len(items), "items": items}
    except Exception as e:
        return _error_reply(f"读取库存失败：{e}")


@app.post("/api/inventory/update")
async def api_inventory_update(req: Request):
    """新增或更新库存 — JSON body: {name, stock, price_per_piece, type?, grade?}"""
    try:
        data = await req.json()
        name = data.get("name", "").strip()
        stock = data.get("stock", 0)
        price_per_piece = data.get("price_per_piece", 0)
        if not name:
            return _error_reply("产品名不能为空")
        item = inventory.add_or_update(
            name=name, stock=stock, price_per_piece=price_per_piece,
            product_type=data.get("type", ""),
            grade=data.get("grade", ""),
        )
        return {"ok": True, "item": item}
    except Exception as e:
        return _error_reply(f"更新库存失败：{e}")


@app.post("/api/inventory/stock-in")
async def api_inventory_stock_in(req: Request):
    """入库 — JSON body: {name, amount?}"""
    try:
        data = await req.json()
        name = data.get("name", "").strip()
        amount = data.get("amount", 1)
        item = inventory.stock_in(name, amount)
        if item is None:
            return _error_reply(f"未找到产品：{name}")
        return {"ok": True, "item": item}
    except Exception as e:
        return _error_reply(f"入库失败：{e}")


@app.post("/api/inventory/stock-out")
async def api_inventory_stock_out(req: Request):
    """出库/售出 — JSON body: {name, amount?}"""
    try:
        data = await req.json()
        name = data.get("name", "").strip()
        amount = data.get("amount", 1)
        item = inventory.stock_out(name, amount)
        if item is None:
            return _error_reply(f"未找到产品：{name}")
        return {"ok": True, "item": item}
    except Exception as e:
        return _error_reply(f"出库失败：{e}")


@app.post("/api/inventory/delete")
async def api_inventory_delete(req: Request):
    """删除珠子品种 — JSON body: {name}"""
    try:
        data = await req.json()
        name = data.get("name", "").strip()
        if not name:
            return _error_reply("产品名不能为空")
        item = inventory.delete(name)
        if item is None:
            return _error_reply(f"未找到产品：{name}")
        return {"ok": True, "item": item}
    except Exception as e:
        return _error_reply(f"删除失败：{e}")


# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
