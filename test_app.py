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
