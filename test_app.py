"""基础测试 — 检索 + API 链路"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# 启动时 rag 模块会自动加载模型和 ChromaDB，需要一点时间
from app import app

client = TestClient(app)


# ============================================================
# 检索测试
# ============================================================
@pytest.mark.asyncio
async def test_search_relevance():
    """search_knowledge('圆脸适合什么') 返回的内容应该和圆脸相关"""
    from rag import hybrid_search

    docs, confidence = await hybrid_search("圆脸适合什么珍珠")

    assert len(docs) > 0, "应该至少返回一条结果"
    # 第一条结果应该包含"圆脸"
    assert "圆脸" in docs[0], f"第一条结果应该和圆脸相关，实际是：{docs[0][:50]}"


@pytest.mark.asyncio
async def test_search_returns_multiple():
    """检索应该返回多条结果"""
    from rag import hybrid_search

    docs, confidence = await hybrid_search("珍珠怎么保养")
    assert len(docs) >= 2, f"应该返回至少 2 条，实际：{len(docs)}"


# ============================================================
# API 测试（Mock DeepSeek 避免真实调用）
# ============================================================
def test_reply_api_endpoint():
    """POST /api/reply 应该返回 200 + reply 字段"""
    with patch("app._llm_client.chat.completions.create") as mock_llm:
        # 模拟 DeepSeek 返回
        mock_choice = MagicMock()
        mock_choice.message.content = "V字型项链很适合圆脸哦～"
        mock_llm.return_value = MagicMock(choices=[mock_choice])

        resp = client.post("/api/reply", json={"question": "圆脸戴什么珍珠"})

    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "sources" in data
    assert len(data["reply"]) > 0


def test_reply_empty_question_rejected():
    """空问题应该被拒绝"""
    resp = client.post("/api/reply", json={"question": ""})
    assert resp.status_code == 422  # Pydantic 校验失败


def test_content_api_endpoint():
    """POST /api/content 应该返回 200 + content 字段"""
    with patch("app._llm_client.chat.completions.create") as mock_llm:
        mock_choice = MagicMock()
        mock_choice.message.content = "**推荐标题：**\n1. 珍珠保养秘籍"
        mock_llm.return_value = MagicMock(choices=[mock_choice])

        resp = client.post("/api/content", json={"topic": "珍珠保养"})

    assert resp.status_code == 200
    data = resp.json()
    assert "content" in data
    assert "sources" in data


def test_content_empty_topic_rejected():
    """空话题应该被拒绝"""
    resp = client.post("/api/content", json={"topic": ""})
    assert resp.status_code == 422


def test_stats_endpoint():
    """GET /api/stats 应该返回统计字段"""
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "copied" in data
    assert "adoption_rate" in data


def test_feedback_endpoint():
    """POST /api/feedback 应该接受有效请求"""
    resp = client.post("/api/feedback", json={
        "action": "copied",
        "question": "测试问题",
        "reply": "测试回复",
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_index_page():
    """首页应该返回 HTML"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "珍珠" in resp.text


# ============================================================
# 库存逻辑测试（mock JSON 文件，不改真实数据）
# ============================================================
def test_stock_in_increases_count():
    """入库后库存数量正确增加"""
    import inventory
    from unittest.mock import patch

    fake_items = [{"name": "测试珠子", "stock": 5, "price_per_piece": 100, "type": "淡水", "grade": "测试"}]

    with patch("inventory._load", return_value=[dict(fake_items[0])]) as mock_load, \
         patch("inventory._save") as mock_save:
        result = inventory.stock_in("测试珠子", amount=3)

    assert result is not None
    assert result["stock"] == 8
    mock_save.assert_called_once()
    # 验证 _save 收到的数据
    saved_data = mock_save.call_args[0][0]
    assert saved_data[0]["stock"] == 8


def test_stock_out_decreases_count():
    """出库后库存正确减少，不会变成负数"""
    import inventory
    from unittest.mock import patch

    fake_items = [{"name": "测试珠子", "stock": 5, "price_per_piece": 100, "type": "淡水", "grade": "测试"}]

    with patch("inventory._load", return_value=[dict(fake_items[0])]), \
         patch("inventory._save") as mock_save:
        result = inventory.stock_out("测试珠子", amount=2)

    assert result is not None
    assert result["stock"] == 3


def test_stock_out_floor_zero():
    """出库超过库存时归零，不出现负数"""
    import inventory
    from unittest.mock import patch

    fake_items = [{"name": "测试珠子", "stock": 3, "price_per_piece": 100, "type": "淡水", "grade": "测试"}]

    with patch("inventory._load", return_value=[dict(fake_items[0])]), \
         patch("inventory._save") as mock_save:
        result = inventory.stock_out("测试珠子", amount=99)

    assert result["stock"] == 0


def test_stock_in_nonexistent_item():
    """入库不存在的产品返回 None"""
    import inventory
    from unittest.mock import patch

    with patch("inventory._load", return_value=[]), \
         patch("inventory._save"):
        result = inventory.stock_in("不存在的珠子", amount=1)

    assert result is None


# ============================================================
# RRF 融合测试（纯函数，不依赖外部资源）
# ============================================================
def test_rrf_fusion_dedup():
    """RRF 融合不会出现重复 doc_id"""
    from rag import rrf_fusion

    bm25 = [("doc_1", "文档A", 8.0), ("doc_2", "文档B", 5.0)]
    vec = [("doc_2", "文档B", 0.85), ("doc_3", "文档C", 0.70)]

    result = rrf_fusion(bm25, vec)

    # 去重：3 个不同文档
    assert len(result) == 3
    assert len(result) == len(set(result)), f"有重复：{result}"


def test_rrf_fusion_ranking():
    """RRF 融合后双路命中的文档排更前"""
    from rag import rrf_fusion

    bm25 = [("doc_1", "文档A", 8.0), ("doc_2", "文档B", 5.0), ("doc_3", "文档C", 3.0)]
    vec = [("doc_2", "文档B", 0.90), ("doc_3", "文档C", 0.85), ("doc_1", "文档A", 0.60)]

    result = rrf_fusion(bm25, vec)

    # doc_2 在两路都排名靠前，应该第一
    assert result[0] == "doc_2", f"双路命中的 doc_2 应该排第一，实际：{result}"


# ============================================================
# 语义记忆测试
# ============================================================
def test_get_relevant_memory_returns_top_k(monkeypatch):
    """语义记忆返回不超过 MEMORY_TOP_K 条"""
    from app import _get_relevant_memory, MEMORY_TOP_K

    # 造 10 条历史
    history = []
    for i in range(10):
        history.append({"question": f"问题{i}", "reply": f"回复{i}"})

    class FakeDB:
        @staticmethod
        def get_conversations(session_id):
            return history

    monkeypatch.setattr("app.db", FakeDB())

    result = _get_relevant_memory("test_session", "珍珠光泽怎么看")
    assert len(result) <= MEMORY_TOP_K


def test_get_relevant_memory_empty_history(monkeypatch):
    """无历史时返回空列表"""
    from app import _get_relevant_memory

    class FakeDB:
        @staticmethod
        def get_conversations(session_id):
            return []

    monkeypatch.setattr("app.db", FakeDB())

    result = _get_relevant_memory("test_session", "随便问")
    assert result == []


# ============================================================
# 预算提取测试 — _extract_budget
# ============================================================
@pytest.mark.parametrize("question,expected", [
    # ── 范围格式：精确返回 ──
    ("预算2000-3000", (2000, 3000)),
    ("预算2000—3000", (2000, 3000)),       # 中文破折号
    ("预算2000–3000", (2000, 3000)),       # en-dash
    ("预算 2000 - 3000", (2000, 3000)),    # 带空格
    ("预算2000到3000", (2000, 3000)),
    ("预算2000至3000", (2000, 3000)),
    ("3000-5000预算", (3000, 5000)),       # 后置预算关键词
    ("2000-3000想买", (2000, 3000)),       # 后置隐式预算
    # ── 上限约束：不超过/以内 → 下限 None ──
    ("预算不超过1000", (None, 1000)),
    ("预算不超过 1000", (None, 1000)),
    ("预算不超1000", (None, 1000)),
    ("预算1000以内", (None, 1000)),
    ("预算1000以下", (None, 1000)),
    # ── 下限约束：以上 → 上限 None ──
    ("预算5000以上", (5000, None)),
    ("预算5000以上有推荐吗", (5000, None)),
    # ── 修饰词：左右/大约/大概/差不多 → ±30% ──
    ("预算2000左右", (1400, 2600)),        # 2000*0.7, 2000*1.3
    ("预算2000左右有推荐吗", (1400, 2600)),
    ("预算3000 左右", (2100, 3900)),
    ("3000 左右的珍珠", (2100, 3900)),
    ("买个珍珠 3000 块左右", (2100, 3900)),
    # ── 兜底：纯预算数字 → ×0.5~×1.1 ──
    ("预算2000", (1000, 2200)),
    ("预算 2000", (1000, 2200)),
    ("预算2000买什么", (1000, 2200)),
    # ── 纯数字表达式 → ×0.5~×1.1 ──
    ("3000块", (1500, 3300)),
    ("3000元", (1500, 3300)),
    ("3000块钱", (1500, 3300)),
])
def test_extract_budget(question, expected):
    """预算提取覆盖各种自然语言表达"""
    from app import _extract_budget
    result = _extract_budget(question)
    assert result is not None, f"应该提取到预算，输入: {question}"
    if expected[0] is None:
        assert result[0] is None, f"下限应为None，输入: {question}，实际: {result[0]}"
    else:
        assert result[0] == pytest.approx(expected[0], rel=0.01), \
            f"下限不对，输入: {question}，期望≈{expected[0]}，实际: {result[0]}"
    if expected[1] is None:
        assert result[1] is None, f"上限应为None，输入: {question}，实际: {result[1]}"
    else:
        assert result[1] == pytest.approx(expected[1], rel=0.01), \
            f"上限不对，输入: {question}，期望≈{expected[1]}，实际: {result[1]}"


def test_extract_budget_no_budget():
    """没有预算信息时返回 (None, None)"""
    from app import _extract_budget
    result = _extract_budget("圆脸适合什么珍珠")
    assert result == (None, None)


# ============================================================
# 预算过滤测试 — _filter_by_budget
# ============================================================
FAKE_INVENTORY = [
    {"name": "淡水珍珠 9-10mm", "price_per_piece": 500, "type": "淡水", "stock": 10},
    {"name": "Akoya珍珠 7-8mm", "price_per_piece": 1200, "type": "海水", "stock": 8},
    {"name": "澳白珍珠 10-11mm", "price_per_piece": 3500, "type": "海水", "stock": 5},
    {"name": "南洋金珠 9-10mm", "price_per_piece": 2800, "type": "海水", "stock": 3},
    {"name": "无价珠子", "price_per_piece": 0, "type": "未知", "stock": 1},
]


def test_filter_by_budget_keeps_items_in_range():
    """预算 1000-3000：应保留 1200 和 2800,过滤掉 500 和 3500"""
    from app import _filter_by_budget
    result = _filter_by_budget(FAKE_INVENTORY, 1000, 3000)
    names = [r["name"] for r in result]
    assert "Akoya珍珠 7-8mm" in names
    assert "南洋金珠 9-10mm" in names
    assert "淡水珍珠 9-10mm" not in names   # 500 < 1000
    assert "澳白珍珠 10-11mm" not in names  # 3500 > 3000*1.1=3300


def test_filter_by_budget_max_elasticity():
    """上限弹性：3000 预算可以买到 3500*1.1=3850 以内的"""
    from app import _filter_by_budget
    # 只有上限，澳白 3500 <= 3000*1.1=3300? No. So 3500 should NOT pass.
    result = _filter_by_budget(FAKE_INVENTORY, None, 3000)
    names = [r["name"] for r in result]
    assert "淡水珍珠 9-10mm" in names       # 500 <= 3300
    assert "Akoya珍珠 7-8mm" in names       # 1200 <= 3300
    assert "南洋金珠 9-10mm" in names       # 2800 <= 3300
    assert "澳白珍珠 10-11mm" not in names  # 3500 > 3300


def test_filter_by_budget_min_floor():
    """下限无弹性：_extract_budget 已做弹性，_filter_by_budget 不下沉"""
    from app import _filter_by_budget
    result = _filter_by_budget(FAKE_INVENTORY, 1000, None)
    names = [r["name"] for r in result]
    assert "淡水珍珠 9-10mm" not in names   # 500 < 1000
    assert "Akoya珍珠 7-8mm" in names       # 1200 >= 1000
    assert "澳白珍珠 10-11mm" in names      # 3500 >= 1000


def test_filter_by_budget_no_limit_returns_all():
    """无预算限制返回全部（含 price=0 的异常数据也行，无限制时不检查价格）"""
    from app import _filter_by_budget
    result = _filter_by_budget(FAKE_INVENTORY, None, None)
    assert len(result) == 5  # 全部返回，不检查 price<=0


# ============================================================
# 年龄→年龄段映射测试 — _age_to_range
# ============================================================
@pytest.mark.parametrize("age,expected_range", [
    (15, "十几岁"),
    (19, "十几岁"),
    (20, "20多岁"),
    (25, "20多岁"),
    (29, "20多岁"),
    (30, "30多岁"),
    (35, "30多岁"),
    (39, "30多岁"),
    (40, "40多岁"),
    (45, "40多岁"),
    (49, "40多岁"),
    (50, "50多岁"),
    (55, "50多岁"),
    (59, "50多岁"),
    (60, "60多岁"),
    (70, "60多岁"),
])
def test_age_to_range(age, expected_range):
    """年龄正确映射到年龄段"""
    from rag import _age_to_range
    assert _age_to_range(age) == expected_range


# ============================================================
# 预算范围扩展测试 — _expand_budget_range
# ============================================================
@pytest.mark.parametrize("query,expected_contains", [
    ("预算6000", "5000-10000"),
    ("预算2800", "2000-3000"),
    ("预算1500", "1000-2000"),
    ("预算800", "1000左右"),
    ("预算400", "500左右"),
    ("预算200", "200-300"),
    ("预算3000", "2000-3000"),      # 边界值，命中 2000-3000 范围
    ("预算1000", "1000左右"),       # 边界值，命中 1000-2000? 不对，1000在700-1400范围 → "1000左右"
    ("预算5000", ["3000-5000", "5000-10000"]),     # 边界值同时命中两个范围
    ("预算1000", ["1000左右", "1000-2000"]),        # 边界值同时命中
])
def test_expand_budget_range(query, expected_contains):
    """预算数字正确映射到知识库范围关键词"""
    from rag import _expand_budget_range
    result = _expand_budget_range(query)
    if isinstance(expected_contains, list):
        for tag in expected_contains:
            assert tag in result, f"输入: {query}，期望包含: {tag}，实际: {result}"
    else:
        assert expected_contains in result, f"输入: {query}，期望包含: {expected_contains}，实际: {result}"


def test_expand_budget_range_no_budget():
    """无预算时原样返回"""
    from rag import _expand_budget_range
    assert _expand_budget_range("圆脸适合什么珍珠") == "圆脸适合什么珍珠"


def test_expand_budget_range_already_has_tag():
    """查询已含范围关键词时不重复追加"""
    from rag import _expand_budget_range
    result = _expand_budget_range("预算6000 5000-10000")
    # 应该不重复追加
    assert result.count("5000-10000") == 1


# ============================================================
# Query 类别提取测试 — _extract_query_categories
# ============================================================
def test_extract_categories_gift():
    """送人 → 用途类"""
    from rag import _extract_query_categories
    cats = _extract_query_categories("买给妈妈 送人")
    assert "用途" in cats


def test_extract_categories_budget_and_scene():
    """预算 + 场景 → 两类"""
    from rag import _extract_query_categories
    cats = _extract_query_categories("预算3000 日常通勤")
    assert "预算" in cats
    assert "场景" in cats


def test_extract_categories_age_face_material():
    """脸型 + 材质 → 两类（"25岁"不直接命中年龄模式，年龄模式匹配的是"20多岁"等范围词）"""
    from rag import _extract_query_categories
    cats = _extract_query_categories("25岁 圆脸 买淡水还是海水")
    # 年龄模式 (十几岁|20多岁|...) 不匹配"25岁"这个具体数字
    assert "脸型" in cats
    assert "材质" in cats


def test_extract_categories_empty():
    """无领域关键词返回空集合"""
    from rag import _extract_query_categories
    cats = _extract_query_categories("你好 谢谢")
    assert cats == set()


def test_extract_categories_shape_quality():
    """形状 + 品质"""
    from rag import _extract_query_categories
    cats = _extract_query_categories("正圆无瑕珍珠")
    assert "形状" in cats
    assert "品质" in cats


# ============================================================
# 知识库关键词提取测试 — _extract_kb_keywords
# ============================================================
def test_extract_kb_keywords_basic():
    """从知识库文档提取珍珠品种"""
    from app import _extract_kb_keywords
    docs = [
        "问：预算1000左右买什么珍珠好？\n答：单颗小澳白是绝对不会出错的",
        "问：预算500左右买什么珍珠好？\n答：可以考虑淡水的单颗吊坠",
    ]
    keywords = _extract_kb_keywords(docs)
    assert "澳白" in keywords
    assert "淡水" in keywords


def test_extract_kb_keywords_jinzhu_unified():
    """金珠 → 统一为南洋金珠"""
    from app import _extract_kb_keywords
    docs = ["金珠是妈妈们的首选，南洋金珠非常贵气"]
    keywords = _extract_kb_keywords(docs)
    assert keywords.count("南洋金珠") == 1  # 金珠和南洋金珠都归为南洋金珠
    assert "金珠" not in keywords


def test_extract_kb_keywords_dedup():
    """去重"""
    from app import _extract_kb_keywords
    docs = [
        "澳白是非常好的选择",
        "澳白的光泽无人能比",
    ]
    keywords = _extract_kb_keywords(docs)
    assert keywords.count("澳白") == 1


def test_extract_kb_keywords_seawater_freshwater():
    """海水珍珠/淡水珍珠提取（长匹配优先）"""
    from app import _extract_kb_keywords
    docs = ["海水珍珠的光泽比淡水珍珠好很多"]
    keywords = _extract_kb_keywords(docs)
    assert "海水珍珠" in keywords
    assert "淡水珍珠" in keywords


# ============================================================
# Redis 降级测试 — Redis 不可用时自动回退内存字典
# ============================================================
def test_cache_fallback_when_redis_down(monkeypatch):
    """Redis 不可用时自动降级内存字典，读写正常不抛异常"""
    import cache

    # 强制 _try_connect_redis 返回 None，模拟 Redis 挂了
    monkeypatch.setattr(cache, "_try_connect_redis", lambda: None)
    # 重置 Redis 客户端缓存（避免之前成功的连接被复用）
    monkeypatch.setattr(cache, "_redis_client", None)

    # 写入
    cache.cache.set("test_fb_key", "test_fb_value", ttl=60)
    # 读取
    result = cache.cache.get("test_fb_key")
    assert result == "test_fb_value", f"降级到内存后应能正常读写，实际: {result}"


def test_cache_fallback_ttl_expires(monkeypatch):
    """内存降级时 TTL 过期后返回 None"""
    import cache
    import time

    monkeypatch.setattr(cache, "_try_connect_redis", lambda: None)
    monkeypatch.setattr(cache, "_redis_client", None)

    # 用已过期的 TTL 写入
    cache.cache.set("test_exp_key", "test_exp_value", ttl=-1)

    # 过期后读取应返回 None
    result = cache.cache.get("test_exp_key")
    assert result is None, f"TTL 过期应返回 None，实际: {result}"


# ============================================================
# 库存原子写入测试 — 写入中断时旧数据完整
# ============================================================
def test_inventory_atomic_write_doesnt_lose_data(tmp_path, monkeypatch):
    """原子写入崩溃场景：临时文件存在但未替换，旧数据完整"""
    import inventory
    import json

    # 在 tmp_path 下创建假的 inventory.json
    inv_file = tmp_path / "inventory.json"
    original_data = [
        {"name": "测试珠子", "stock": 10, "price_per_piece": 100, "type": "淡水", "grade": "测试"}
    ]
    inv_file.write_text(json.dumps(original_data, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(inventory, "INVENTORY_FILE", inv_file)

    # 模拟崩溃：写完 tmp 文件后，replace 还没来得及执行就挂了
    # → tmp 文件存在，但 .json 文件还是旧的
    tmp_file = inv_file.with_suffix(".tmp")
    tmp_file.write_text(
        json.dumps([{"name": "坏数据", "stock": 0}], ensure_ascii=False),
        encoding="utf-8",
    )

    # _load 应该只读 .json 文件，不碰 .tmp
    loaded = inventory._load()
    assert loaded == original_data, (
        f"即使有残留 tmp 文件，也应读取完整旧数据。期望: {original_data}, 实际: {loaded}"
    )

    # 清理
    tmp_file.unlink()


def test_inventory_atomic_write_normal_flow(tmp_path, monkeypatch):
    """正常写入：_save 后 _load 读到新数据"""
    import inventory
    import json

    inv_file = tmp_path / "inventory.json"
    original_data = [
        {"name": "旧珠子", "stock": 5, "price_per_piece": 50, "type": "淡水", "grade": ""}
    ]
    inv_file.write_text(json.dumps(original_data, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(inventory, "INVENTORY_FILE", inv_file)

    # 正常保存新数据
    new_data = [
        {"name": "新珠子", "stock": 20, "price_per_piece": 200, "type": "海水", "grade": "AAA"}
    ]
    inventory._save(new_data)

    # 读取验证
    loaded = inventory._load()
    assert loaded == new_data, f"正常写入后应读到新数据。期望: {new_data}, 实际: {loaded}"

    # 确认 tmp 文件已被 replace 清理（不存在）
    tmp_file = inv_file.with_suffix(".tmp")
    assert not tmp_file.exists(), "正常写入后 tmp 文件应不存在"
