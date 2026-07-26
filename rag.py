"""RAG 检索链路 — BM25 + 向量 → RRF 融合 → DeepSeek Rerank

注意：数据库不在 import 时加载，而是由 app.py 通过 lifespan 调用 initialize()。
Embedding 改为 API 调用（不再本地加载 SentenceTransformer 模型），内存占用从 1.2GB → ~200MB。
"""
import os
import asyncio
import re
import json
import time
import logging
import threading
import shutil
import requests
from datetime import datetime, timezone
import jieba
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from pathlib import Path
from openai import OpenAI

from prompts import RERANK_PROMPT
from cache import cache

# ---- Embedding API 配置（通义千问原生 API） ----
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")

# ---- 检索专用日志 ----
retrieval_logger = logging.getLogger("pearl.retrieval")
retrieval_logger.setLevel(logging.DEBUG)
if not retrieval_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[检索] %(message)s"))
    retrieval_logger.addHandler(_h)
    retrieval_logger.propagate = False

# ---- 检索日志持久化（JSONL，供 bad case 分析） ----
RETRIEVAL_LOG_FILE = Path(__file__).parent / "retrieval_log.jsonl"
_retrieval_log_lock = threading.Lock()

# 向量相似度阈值：最佳匹配低于此值视为"知识库没有相关知识"
SIMILARITY_FLOOR = 0.35

# Query 改写：短于此长度的输入自动触发改写
REWRITE_MIN_LENGTH = 8

# ============================================================
# 常量
# ============================================================
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "pearl_knowledge"
# EMBEDDING_MODEL_NAME 已在文件顶部从环境变量读取（默认 text-embedding-v2）

# ============================================================
# LangChain 适配：DashScope Embedding API → LangChain embedding 接口
# ============================================================
class _APIEmbeddings:
    """把 DashScope Embedding API 包装成 LangChain 兼容的 embedding 接口。
    使用通义千问原生 API（不走 OpenAI 兼容模式），不再本地加载模型。
    """

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """调通义千问原生文本 Embedding API"""
        if not texts:
            return []
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": {"texts": texts}},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        # 按 text_index 排序保证顺序不变
        items = sorted(data["output"]["embeddings"], key=lambda e: e["text_index"])
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call_api(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text])[0]


# ============================================================
# 懒加载状态 — initialize() 调用前全部为 None / 空
# ============================================================
_vectorstore: Chroma | None = None
_all_ids: list[str] = []
_all_documents: list[str] = []
_id_to_doc: dict[str, str] = {}
_doc_to_id: dict[str, str] = {}
_bm25: BM25Okapi | None = None
_llm_client: OpenAI | None = None
_initialized: bool = False


# ============================================================
# 初始化（由 app.py 的 lifespan 调用，不再在 import 时执行）
# ============================================================
def _migrate_chroma_to_api(embedding_fn: _APIEmbeddings):
    """一次性迁移：把旧版 SentenceTransformer 向量库重建为 API embedding 向量库。
    用 chromadb 自身 API 删旧集合（避免 Windows 文件锁），再用 LangChain Chroma 重建。
    """
    version_file = CHROMA_DIR / "embedding_version.txt"
    if version_file.exists():
        return  # 已是 API 版本，无需迁移

    docs: list[str] = []
    ids: list[str] = []

    # 尝试从旧版 chroma_db 提取文档
    try:
        import chromadb
        from chromadb.config import Settings as CDBSettings
        old_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=CDBSettings(anonymized_telemetry=False),
        )
        old_col = old_client.get_collection(COLLECTION_NAME)
        data = old_col.get(include=["documents"])
        docs = list(data.get("documents", []))
        ids = list(data.get("ids", []))
        if docs:
            print(f"从旧版向量库提取了 {len(docs)} 条文档，开始迁移...")
            # 用 chromadb API 删旧集合
            old_client.delete_collection(COLLECTION_NAME)
        del old_col
        del old_client
        # 清 chromadb 内部单例缓存，否则 LangChain Chroma 打不开同目录
        import gc
        gc.collect()
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient._identifier_to_system.pop(str(CHROMA_DIR), None)
        except Exception:
            pass
    except Exception as e:
        print(f"旧版向量库读取失败: {e}")
        try:
            del old_client
        except Exception:
            pass

    # 只有成功提取到文档才重建
    if not docs:
        print("未提取到旧文档，保留现有向量库，标记为 API 版本")
        version_file.write_text("api-v2")
        return

    # 用 LangChain Chroma 创建新集合（API embedding）
    new_vs = Chroma(
        embedding_function=embedding_fn,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"正在用 API embedding 重建向量库（{len(docs)} 条文档）...")
    new_vs.add_texts(texts=docs, ids=ids)
    print("向量库重建完成")
    version_file.write_text("api-v2")


def initialize():
    """通过 LangChain Chroma 连接向量库、构建 BM25 索引。
    必须在服务启动时调用一次，否则所有检索函数会抛出 RuntimeError。

    Embedding 改为 API 调用（DashScope text-embedding-v2），不再本地加载模型。
    如果 chroma_db 是旧版 SentenceTransformer 构建的，自动迁移。
    """
    global _vectorstore
    global _all_ids, _all_documents, _id_to_doc, _doc_to_id, _bm25, _initialized

    embedding_fn = _APIEmbeddings(EMBEDDING_API_KEY, EMBEDDING_MODEL_NAME)

    # 自动迁移旧版向量库（仅首次）
    _migrate_chroma_to_api(embedding_fn)

    # ⚠️ 清 chromadb 单例缓存——迁移函数里的旧 PersistentClient 可能还占着锁
    import gc
    gc.collect()
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient._identifier_to_system.clear()
    except Exception:
        pass

    # 用 LangChain Chroma 封装
    _vectorstore = Chroma(
        embedding_function=embedding_fn,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    # 提取全量文档用于 BM25 和 Rerank
    all_data = _vectorstore.get(include=["documents"])
    _all_ids = list(all_data.get("ids", []))
    _all_documents = list(all_data.get("documents", []))
    _id_to_doc = dict(zip(_all_ids, _all_documents))
    _doc_to_id = dict(zip(_all_documents, _all_ids))

    tokenized_corpus = [list(jieba.cut(doc)) for doc in _all_documents]
    _bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    _initialized = True
    print(f"知识库就绪 — {len(_all_documents)} 条文档（API Embedding: {EMBEDDING_MODEL_NAME}）")


def _ensure_initialized():
    """所有检索函数的统一守卫"""
    if not _initialized:
        raise RuntimeError("知识库尚未初始化，请等待服务启动完成")


def set_llm_client(client: OpenAI):
    """由 app.py 注入 DeepSeek 客户端"""
    global _llm_client
    _llm_client = client


# ============================================================
# 检索指标追踪
# ============================================================
class RetrievalMetrics:
    """检索链路指标（线程安全）。供 /api/metrics 查询。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_searches = 0
        self.low_confidence_count = 0
        self.rewrite_count = 0
        self.rerank_count = 0
        self.rerank_fallback_count = 0
        self.sum_max_similarity = 0.0
        self.sum_latency_ms = 0.0

    def record(self, *, max_sim: float, latency_ms: float,
               was_rewritten: bool = False, was_low_confidence: bool = False,
               rerank_used: bool = False, rerank_fell_back: bool = False):
        with self._lock:
            self.total_searches += 1
            self.sum_max_similarity += max_sim
            self.sum_latency_ms += latency_ms
            if was_rewritten:
                self.rewrite_count += 1
            if was_low_confidence:
                self.low_confidence_count += 1
            if rerank_used:
                self.rerank_count += 1
            if rerank_fell_back:
                self.rerank_fallback_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            n = self.total_searches or 1
            return {
                "total_searches": self.total_searches,
                "avg_similarity": round(self.sum_max_similarity / n, 3),
                "avg_latency_ms": round(self.sum_latency_ms / n, 1),
                "low_confidence_rate": round(self.low_confidence_count / n, 3),
                "rewrite_rate": round(self.rewrite_count / n, 3),
                "rerank_used": self.rerank_count,
                "rerank_fallback_rate": (
                    round(self.rerank_fallback_count / max(self.rerank_count, 1), 3)
                ),
            }


_metrics = RetrievalMetrics()


def get_retrieval_metrics() -> dict:
    """导出检索指标快照（app.py 调用）"""
    return _metrics.snapshot()


# ============================================================
# Query 改写 — 口语/短输入自动补全
# ============================================================
REWRITE_SYSTEM_PROMPT = """你是珍珠行业的检索助手。客户输入可能很口语、很短、很模糊。
你的任务是把客户输入改写成适合在珍珠知识库里检索的查询语句。

改写规则：
1. 补全省略的主语/宾语（比如"好看的"→"好看的珍珠项链推荐"）
2. 口语转书面（比如"亮不亮"→"珍珠光泽好不好"）
3. 保持简洁，不要超过20个字
4. 如果原文已经足够清晰完整，直接输出原文
5. 只输出改写后的查询，不要任何解释"""


# 年龄数字→年龄段映射，让 "25岁" 能匹配知识库里的 "20多岁年轻女生"
_AGE_EXPAND = [
    (re.compile(r'(\d{2})岁'), lambda age: _age_to_range(int(age))),
]

def _age_to_range(age: int) -> str:
    """把具体年龄映射到知识库里的年龄段表述（只加一个词，避免淹没其他关键词）"""
    if age < 20:
        return "十几岁"
    elif age < 30:
        return "20多岁"
    elif age < 40:
        return "30多岁"
    elif age < 50:
        return "40多岁"
    elif age < 60:
        return "50多岁"
    else:
        return "60多岁"

# 预算数字→知识库范围映射，让 "6000" 能匹配 "预算5000-10000"
_BUDGET_RANGE_MAP = [
    # (min, max, keyword_to_append)
    (150, 350, "200-300"),
    (350, 700, "500左右"),
    (700, 1400, "1000左右"),
    (1000, 2000, "1000-2000"),
    (2000, 3000, "2000-3000"),
    (3000, 5000, "3000-5000"),
    (5000, 10000, "5000-10000"),
]

def _expand_budget_range(query: str) -> str:
    """输入 '预算6000' →检测数字6000在5000-10000范围→追加'5000-10000'

    边界值（如 5000 同时在 3000-5000 和 5000-10000 的边界）→ 追加两个标签。
    """
    budget_pat = re.compile(r'预算\s*(\d+)')
    m = budget_pat.search(query)
    if not m:
        return query
    budget = int(m.group(1))
    matched_tags = []
    for lo, hi, tag in _BUDGET_RANGE_MAP:
        if lo <= budget <= hi:
            if tag not in query:
                matched_tags.append(tag)
    if matched_tags:
        retrieval_logger.info("✏️ 预算扩展: %d元 → %s", budget, matched_tags)
        return f"{query} {' '.join(matched_tags)}"
    return query


def _expand_age_range(query: str) -> str:
    """检测 query 中的年龄数字，追加对应的年龄段关键词"""
    for pat, mapper in _AGE_EXPAND:
        m = pat.search(query)
        if m:
            age = int(m.group(1))
            if 10 <= age <= 99:  # 合理年龄范围
                expansion = mapper(age)
                if expansion not in query:
                    retrieval_logger.info("✏️ 年龄扩展: %d岁 → '%s'", age, expansion)
                    return f"{query} {expansion}"
    return query


async def _maybe_rewrite_query(query: str) -> tuple[str, bool]:
    """短/口语查询自动改写。返回 (rewritten_query, was_rewritten)。"""
    # 长度够且包含明确关键词 → 跳过改写
    if len(query) >= REWRITE_MIN_LENGTH:
        return query, False
    # 纯数字/符号 → 不改
    if not any('一' <= c <= '鿿' for c in query):
        return query, False

    if _llm_client is None:
        return query, False

    try:
        resp = await asyncio.to_thread(
            lambda: _llm_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"改写：{query}"},
                ],
                temperature=0.0,
                max_tokens=40,
            )
        )
        rewritten = resp.choices[0].message.content.strip()
        if rewritten and rewritten != query and len(rewritten) > 1:
            retrieval_logger.info("✏️ Query 改写: '%s' → '%s'", query, rewritten)
            return rewritten, True
    except Exception:
        pass
    return query, False


# ============================================================
# 检索日志持久化
# ============================================================
def _persist_retrieval_log(entry: dict):
    """追加一条结构化检索日志到 JSONL 文件"""
    try:
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        with _retrieval_log_lock:
            with open(RETRIEVAL_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ============================================================
# 检索函数
# ============================================================
def bm25_search(query: str, top_k: int = 15) -> list[tuple[str, str, float]]:
    """BM25 关键词检索。返回 [(doc_id, doc_content, score), ...]"""
    _ensure_initialized()
    if not _bm25:
        return []
    tokens = list(jieba.cut(query))
    scores = _bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        (_all_ids[i], _all_documents[i], float(scores[i]))
        for i, score in ranked
    ]


async def vector_search(query: str, top_k: int = 15) -> list[tuple[str, str, float]]:
    """向量语义检索（通过 LangChain Chroma，异步）。返回 [(doc_id, doc_content, similarity), ...]"""
    _ensure_initialized()
    # LangChain Chroma 的 similarity_search_with_score 内部处理了 embedding
    results = await asyncio.to_thread(
        _vectorstore.similarity_search_with_score, query, k=top_k
    )
    output: list[tuple[str, str, float]] = []
    for doc, score in results:
        # Chroma cosine 距离 → 相似度
        sim = 1.0 - score
        doc_id = _doc_to_id.get(doc.page_content, "")
        if doc_id:
            output.append((doc_id, doc.page_content, sim))
    return output


def rrf_fusion(
    bm25_results: list[tuple[str, str, float]],
    vector_results: list[tuple[str, str, float]],
    k: int = 60,
) -> list[str]:
    """RRF 融合 — 按 doc_id 去重，合并 BM25 和向量排序"""
    scores: dict[str, float] = {}
    for rank, (doc_id, _doc, _score) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _doc, _score) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


# ---- 领域关键词词典：提取 query 中的强约束信号 ----
_DOMAIN_PATTERNS: list[tuple[str, str]] = [
    # (正则, 类别名)
    (r"(十几岁|20多岁|30多岁|40多岁|50多岁|60多岁)", "年龄"),
    (r"预算\s*(\d+)\s*(块|元|百|千|万|以内|左右|上下)?", "预算"),
    (r"(\d+)\s*(块|元|百|千|万|以内|左右|上下|预算)", "预算"),
    (r"(海水|淡水|Akoya|akoya|大溪地|南洋金珠|马贝|爱迪生|巴洛克|keshi)", "材质"),
    (r"(圆脸|长脸|方脸|瓜子脸|鹅蛋脸|国字脸|显脸|脸型|脸大)", "脸型"),
    (r"(送妈妈|送长辈|送女友|送闺蜜|送婆婆|送岳母|买给妈妈|给妈妈|送人|生日|结婚|婚礼|礼物|本命年)", "用途"),
    # 预算扩展标签（_expand_budget_range 生成的关键词）
    (r"(200-300|500左右|1000左右|1000-2000|2000-3000|3000-5000|5000-10000)", "预算"),
    (r"(日常|上班|通勤|百搭|配衣服|正式|职场|休闲)", "场景"),
    (r"(7-8mm|8-9mm|9-10mm|10mm|12mm|小米珠|小尺寸|大尺寸|点位|直径)", "尺寸"),
    (r"(正圆|近圆|水滴|馒头|椭圆|异形|螺纹)", "形状"),
    (r"(无瑕|极微瑕|微瑕|少瑕|瑕疵|光洁度)", "品质"),
]


def _extract_query_categories(query: str) -> set[str]:
    """提取 query 命中的领域类别（不是具体关键词，是类别标签）。

    比如 "买给妈妈 送人" → {"用途"}，"预算3000 日常" → {"预算", "场景"}。
    """
    categories: set[str] = set()
    for pattern, category in _DOMAIN_PATTERNS:
        if re.search(pattern, query):
            categories.add(category)
    return categories


def _keyword_aware_rerank(query: str, doc_ids: list[str]) -> list[str]:
    """领域关键词加权重排：类别匹配 + 精确标签匹配双层加分。

    第一层（类别级）：query "送人" → 类别"用途" → doc "送长辈婆婆岳母" 含"送长辈" → +1
    第二层（精确级）：query 含扩展标签"2000-3000" → doc 含"2000-3000" → 额外 +2

    精确级加分确保预算/年龄范围精准命中时优先于同类但范围不同的文档。
    比如 query 预算 2500 → 扩展为"2000-3000" → doc 含"2000-3000"的排名高于"1000左右"。
    """
    query_categories = _extract_query_categories(query)
    if not query_categories or len(doc_ids) <= 3:
        return doc_ids

    # 提取 query 中的精确标签（预算范围/年龄段等数字关键词）
    exact_tags: set[str] = set()
    for pattern, _category in _DOMAIN_PATTERNS:
        for m in re.finditer(pattern, query):
            tag = m.group(0)
            if len(tag) >= 3:  # 过滤太短的匹配
                exact_tags.add(tag)

    boost: dict[str, int] = {}
    for doc_id in doc_ids:
        doc_text = _id_to_doc.get(doc_id, "")
        score = 0
        # 类别级 +1
        for pattern, category in _DOMAIN_PATTERNS:
            if category in query_categories and re.search(pattern, doc_text):
                score += 1
        # 精确标签级 +2（预算范围/年龄段等数字标签）
        for tag in exact_tags:
            if tag in doc_text:
                score += 2
        boost[doc_id] = score

    retrieval_logger.info(
        "关键词预过滤 — query 命中类别: %s, 精确标签: %s", query_categories, exact_tags,
    )
    for doc_id in doc_ids[:8]:
        doc_text = _id_to_doc.get(doc_id, "")[:80]
        retrieval_logger.info(
            "  boost=%d doc=%s", boost.get(doc_id, 0), doc_text,
        )

    return sorted(doc_ids, key=lambda did: boost.get(did, 0), reverse=True)


def rerank_with_deepseek(query: str, candidate_ids: list[str], top_k: int = 3) -> list[str]:
    """用 DeepSeek 做 Rerank：让模型自己挑最相关的 top_k 条"""
    if len(candidate_ids) <= top_k:
        return candidate_ids

    candidates = [_id_to_doc.get(doc_id, "") for doc_id in candidate_ids]
    docs_text = "\n\n---\n\n".join(
        f"[文档 {i+1}]\n{doc}" for i, doc in enumerate(candidates)
    )
    prompt = RERANK_PROMPT.format(query=query, top_k=top_k, docs_text=docs_text)

    try:
        resp = _llm_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )
        result_text = resp.choices[0].message.content.strip()
        numbers = re.findall(r"\d+", result_text)
        ranked_ids = []
        seen = set()
        for n in numbers:
            idx = int(n) - 1
            if 0 <= idx < len(candidate_ids) and candidate_ids[idx] not in seen:
                ranked_ids.append(candidate_ids[idx])
                seen.add(candidate_ids[idx])
        return ranked_ids[:top_k] if ranked_ids else candidate_ids[:top_k]
    except Exception:
        return candidate_ids[:top_k]


async def hybrid_search(query: str, top_k: int = 3) -> tuple[list[str], float]:
    """混合检索完整流程：Query改写 → BM25 + 向量 → RRF 融合 → DeepSeek Rerank

    返回 (documents, confidence)：
    - documents: 检索到的文档内容列表
    - confidence: 0.0~1.0，最高向量相似度，低于 SIMILARITY_FLOOR 说明知识库不匹配
    """
    _ensure_initialized()
    t0 = time.perf_counter()

    # ---- Redis 缓存命中 → 直接返回 ----
    import hashlib
    cache_key = f"rag:{hashlib.md5(query.encode()).hexdigest()}:{top_k}"
    cached = cache.get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            retrieval_logger.info("⚡ 缓存命中 — query=%s", query[:60])
            latency_ms = (time.perf_counter() - t0) * 1000
            _metrics.record(max_sim=data.get("confidence", 0.0), latency_ms=latency_ms)
            return data["docs"], data["confidence"]
        except (json.JSONDecodeError, KeyError):
            pass  # 缓存数据损坏，正常检索

    # ---- Query 改写 ----
    rewritten_query, was_rewritten = await _maybe_rewrite_query(query)
    search_query = rewritten_query if was_rewritten else query

    # ---- 年龄扩展 + 预算范围扩展 ----
    search_query = _expand_age_range(search_query)
    search_query = _expand_budget_range(search_query)

    retrieval_logger.info("── 检索开始 ── query=%s", search_query[:80])

    # ---- 小知识库模式 ----
    if len(_all_documents) < 5:
        vec_results = await vector_search(search_query, top_k)
        max_sim = max((s for _, _, s in vec_results), default=0.0)
        docs = [doc for _id, doc, _score in vec_results]
        retrieval_logger.info("小库模式 top-%d sim=%.3f", top_k, max_sim)
        for i, d in enumerate(docs):
            retrieval_logger.info("  最终[%d] (sim=%.3f) %s", i + 1, max_sim, d[:100])

        latency_ms = (time.perf_counter() - t0) * 1000
        is_low = max_sim < SIMILARITY_FLOOR
        _metrics.record(max_sim=max_sim, latency_ms=latency_ms,
                        was_rewritten=was_rewritten, was_low_confidence=is_low)
        _persist_retrieval_log({
            "original_query": query,
            "rewritten_query": search_query if was_rewritten else None,
            "top_doc_ids": [],
            "top_similarities": [],
            "max_sim": round(max_sim, 4),
            "low_confidence": is_low,
            "mode": "small_kb",
            "latency_ms": round(latency_ms, 1),
        })
        cache.set(cache_key, json.dumps({"docs": docs, "confidence": max_sim}), ttl=300)
        return docs, max_sim

    # ---- BM25 关键词检索 ----
    bm25_results = bm25_search(search_query, top_k=15)
    retrieval_logger.info("BM25 top-15:")
    for i, (_id, doc, score) in enumerate(bm25_results):
        retrieval_logger.info("  [%d] (bm25=%.2f) %s", i + 1, score, doc[:100])

    # ---- 向量语义检索 ----
    vector_results = await vector_search(search_query, top_k=15)
    max_vec = max((s for _, _, s in vector_results), default=0.0)

    # ---- 双路置信度：向量 + BM25 归一化，取最高分 ----
    max_bm25_raw = max((s for _, _, s in bm25_results), default=0.0)
    bm25_norm = min(max_bm25_raw / 8.0, 1.0) if max_bm25_raw > 0 else 0.0
    max_sim = max(max_vec, bm25_norm)

    retrieval_logger.info("向量 top-15 (max_vec=%.3f, bm25_norm=%.3f, combined=%.3f):",
                          max_vec, bm25_norm, max_sim)
    for i, (_id, doc, sim) in enumerate(vector_results):
        retrieval_logger.info("  [%d] (sim=%.3f) %s", i + 1, sim, doc[:100])

    # ---- RRF 融合 ----
    fused_ids = rrf_fusion(bm25_results, vector_results)
    retrieval_logger.info("RRF 融合 top-10 doc_ids: %s", fused_ids[:10])

    # ---- 领域关键词加权重排：提取预算/脸型/材质/用途，匹配文档加分 ----
    fused_ids = _keyword_aware_rerank(search_query, fused_ids)

    # ---- DeepSeek Rerank ----
    candidate_ids = fused_ids[: min(20, len(fused_ids))]
    rerank_used = len(candidate_ids) > top_k
    rerank_fell_back = False
    ranked_ids = rerank_with_deepseek(search_query, candidate_ids, top_k)
    if rerank_used and ranked_ids == candidate_ids[:top_k]:
        rerank_fell_back = True  # rerank 异常回退到原始顺序

    retrieval_logger.info("Rerank 最终 top-%d:", top_k)
    top_sims = []
    for i, doc_id in enumerate(ranked_ids):
        doc_text = _id_to_doc.get(doc_id, "")
        # 反查这条 doc 的向量相似度（用于日志）
        doc_sim = next((s for _id2, _d2, s in vector_results if _id2 == doc_id), 0.0)
        top_sims.append(round(doc_sim, 4))
        retrieval_logger.info("  最终[%d] %s", i + 1, doc_text[:120])

    docs = [_id_to_doc.get(doc_id, "") for doc_id in ranked_ids]

    # ---- 指标 + 日志 ----
    latency_ms = (time.perf_counter() - t0) * 1000
    is_low = max_sim < SIMILARITY_FLOOR
    _metrics.record(max_sim=max_sim, latency_ms=latency_ms,
                    was_rewritten=was_rewritten, was_low_confidence=is_low,
                    rerank_used=rerank_used, rerank_fell_back=rerank_fell_back)
    _persist_retrieval_log({
        "original_query": query,
        "rewritten_query": search_query if was_rewritten else None,
        "top_doc_ids": ranked_ids[:top_k],
        "top_similarities": top_sims,
        "max_sim": round(max_sim, 4),
        "low_confidence": is_low,
        "mode": "full",
        "latency_ms": round(latency_ms, 1),
    })

    cache.set(cache_key, json.dumps({"docs": docs, "confidence": max_sim}), ttl=300)
    return docs, max_sim
