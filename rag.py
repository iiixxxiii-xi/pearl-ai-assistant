"""RAG 检索链路 — BM25 + 向量 → RRF 融合 → DeepSeek Rerank

注意：模型和数据库不在 import 时加载，而是由 app.py 通过 lifespan 调用 initialize()。
这样 import 不会阻塞，测试也能在不加载模型的情况下 mock 整个模块。
"""
import os

# ⚠️ 必须在 import sentence_transformers 之前设置，否则走的还是 huggingface.co
#    setdefault 不够——如果系统或 .env 里已有值会跳过，这里直接显式覆盖
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import asyncio
import re
import json
import time
import logging
import threading
from datetime import datetime, timezone
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
from openai import OpenAI

from prompts import RERANK_PROMPT
from cache import cache

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
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================
# 懒加载状态 — initialize() 调用前全部为 None / 空
# ============================================================
_embedding_model: SentenceTransformer | None = None
_collection = None
_all_ids: list[str] = []
_all_documents: list[str] = []
_id_to_doc: dict[str, str] = {}
_bm25: BM25Okapi | None = None
_llm_client: OpenAI | None = None
_initialized: bool = False


# ============================================================
# 初始化（由 app.py 的 lifespan 调用，不再在 import 时执行）
# ============================================================
def initialize():
    """加载 Embedding 模型、连接 ChromaDB、构建 BM25 索引。
    必须在服务启动时调用一次，否则所有检索函数会抛出 RuntimeError。
    """
    global _embedding_model, _collection
    global _all_ids, _all_documents, _id_to_doc, _bm25, _initialized

    print(f"正在加载 Embedding 模型: {EMBEDDING_MODEL} ...")
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    _chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _chroma_client.get_collection(name=COLLECTION_NAME)

    all_data = _collection.get()
    _all_ids = list(all_data.get("ids", []))
    _all_documents = list(all_data.get("documents", []))
    _id_to_doc = dict(zip(_all_ids, _all_documents))

    tokenized_corpus = [list(jieba.cut(doc)) for doc in _all_documents]
    _bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    _initialized = True
    print(f"知识库就绪 — {len(_all_documents)} 条文档，BM25 索引已构建")


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
                model="deepseek-chat",
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
    """向量语义检索（异步）。返回 [(doc_id, doc_content, similarity), ...]"""
    _ensure_initialized()
    query_embedding = await asyncio.to_thread(
        _embedding_model.encode, [query]
    )
    results = _collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return [
        (id_, doc, 1.0 - dist)
        for id_, doc, dist in zip(ids, docs, distances)
    ]


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
    (r"预算\s*(\d+)\s*(块|元|百|千|万|以内|左右|上下)?", "预算"),
    (r"(\d+)\s*(块|元|百|千|万|以内|左右|上下|预算)", "预算"),
    (r"(海水|淡水|Akoya|akoya|大溪地|南洋金珠|马贝|爱迪生|巴洛克|keshi)", "材质"),
    (r"(圆脸|长脸|方脸|瓜子脸|鹅蛋脸|国字脸|显脸|脸型|脸大)", "脸型"),
    (r"(送妈妈|送长辈|送女友|送闺蜜|送婆婆|送岳母|生日|结婚|婚礼|礼物|本命年)", "用途"),
    (r"(日常|上班|通勤|百搭|配衣服|正式|职场|休闲)", "场景"),
    (r"(7-8mm|8-9mm|9-10mm|10mm|12mm|小米珠|小尺寸|大尺寸|点位|直径)", "尺寸"),
    (r"(正圆|近圆|水滴|馒头|椭圆|异形|螺纹)", "形状"),
    (r"(无瑕|极微瑕|微瑕|少瑕|瑕疵|光洁度)", "品质"),
]


def _extract_domain_keywords(query: str) -> list[str]:
    """从查询中提取领域关键词（预算数字、材质、脸型等强信号词）。
    用 finditer 而非 search——同一模式可能命中多次（如"大溪地"和"生日"都匹配用途）。"""
    keywords: list[str] = []
    for pattern, _category in _DOMAIN_PATTERNS:
        for m in re.finditer(pattern, query):
            keywords.append(m.group(0))
    return keywords


def _keyword_aware_rerank(query: str, doc_ids: list[str]) -> list[str]:
    """领域关键词加权重排：匹配到关键词的文档排名提至前方。

    不改变 doc_ids 的去重和内容——只在 RRF 排序后、DeepSeek Rerank 前
    做轻量调整。文档标题/首行命中了 query 关键词 → 排名前移。
    """
    keywords = _extract_domain_keywords(query)
    if not keywords or len(doc_ids) <= 3:
        return doc_ids  # 没有领域关键词，不干预

    boost: dict[str, int] = {}
    for doc_id in doc_ids:
        doc_text = _id_to_doc.get(doc_id, "")
        boost[doc_id] = sum(1 for kw in keywords if kw.lower() in doc_text.lower())

    retrieval_logger.info(
        "关键词预过滤 — query 关键词: %s", keywords,
    )
    for doc_id in doc_ids[:8]:
        doc_text = _id_to_doc.get(doc_id, "")[:60]
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
            model="deepseek-chat",
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
