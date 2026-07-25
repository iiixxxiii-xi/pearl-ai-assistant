"""库存管理 — JSON 持久化，提供搜索、增删改、入库/出库

和知识库分离：知识库是"珍珠怎么选"，库存是"有什么珠子可以卖"。
每次检索时并行查库存 + 款式，LLM 生成回复时只推荐有库存的珠子。
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

INVENTORY_FILE = Path(__file__).parent / "data" / "inventory.json"
_lock_path = Path(__file__).parent / "data" / "inventory.lock"

# ============================================================
# 跨进程文件锁 — 多 worker 并发写不会互相覆盖
# ============================================================
if sys.platform == "win32":
    import msvcrt

    @contextmanager
    def _file_lock():
        _lock_path.parent.mkdir(parents=True, exist_ok=True)
        _lock_path.touch(exist_ok=True)
        fd = os.open(str(_lock_path), os.O_RDWR)
        for _ in range(100):
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.01)
        else:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            os.close(fd)

else:
    import fcntl

    @contextmanager
    def _file_lock():
        _lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(_lock_path), "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ============================================================
# 库存 — 珠子原料
# ============================================================

def _load() -> list[dict]:
    """读取库存文件（文件锁保护，确保不读到半写入的脏数据）"""
    if not INVENTORY_FILE.exists():
        return []
    with _file_lock():
        with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def _save(items: list[dict]):
    """原子写入库存文件（临时文件 + rename，跨进程文件锁保护）"""
    INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock():
        tmp = INVENTORY_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        tmp.replace(INVENTORY_FILE)  # 同文件系统上原子替换


def search(query: str, top_k: int = 6) -> list[dict]:
    """关键词搜索库存 — 匹配珠子名称/类型/等级

    返回: [{name, type, price_per_piece, stock, grade}, ...]
    只返回有库存的（stock > 0）
    """
    items = _load()
    results = []

    # 中文用 jieba 分词，英文用空格分词
    try:
        import jieba
        words = list(jieba.cut(query))
    except ImportError:
        words = query.lower().split()

    for item in items:
        if item.get("stock", 0) <= 0:
            continue
        text = f"{item.get('name', '')} {item.get('type', '')} {item.get('grade', '')}".lower()
        score = sum(1 for word in words if word.strip().lower() in text)
        if score > 0:
            results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in results[:top_k]]


def search_by_name(name: str) -> Optional[dict]:
    """按产品名精确查找"""
    items = _load()
    for item in items:
        if item.get("name", "") == name:
            return item
    return None


def add_or_update(name: str, stock: int, price_per_piece: float,
                  product_type: str = "", grade: str = ""):
    """新增或更新库存"""
    items = _load()
    for item in items:
        if item.get("name", "") == name:
            item["stock"] = stock
            item["price_per_piece"] = price_per_piece
            if product_type:
                item["type"] = product_type
            if grade:
                item["grade"] = grade
            _save(items)
            return item

    new_item = {
        "name": name,
        "stock": stock,
        "price_per_piece": price_per_piece,
        "type": product_type,
        "grade": grade,
    }
    items.append(new_item)
    _save(items)
    return new_item


def stock_in(name: str, amount: int = 1):
    """入库 — 增加库存数量"""
    items = _load()
    for item in items:
        if item.get("name", "") == name:
            item["stock"] = item.get("stock", 0) + amount
            _save(items)
            return item
    return None


def delete(name: str) -> Optional[dict]:
    """删除一个珠子品种（确认后不可恢复）"""
    items = _load()
    for i, item in enumerate(items):
        if item.get("name", "") == name:
            deleted = items.pop(i)
            _save(items)
            return deleted
    return None


def stock_out(name: str, amount: int = 1):
    """出库/售出 — 减少库存数量"""
    items = _load()
    for item in items:
        if item.get("name", "") == name:
            item["stock"] = max(0, item.get("stock", 0) - amount)
            _save(items)
            return item
    return None


def list_all() -> list[dict]:
    """列出全部库存"""
    return _load()


def format_inventory_for_prompt(items: list[dict]) -> str:
    """把珠子库存转成 LLM 可读文本"""
    if not items:
        return "（当前没有匹配的珠子库存）"

    lines = ["📦 可售珍珠（单价/颗）："]
    for i, item in enumerate(items, 1):
        stock = item.get("stock", 0)
        name = item.get("name", "")
        price = item.get("price_per_piece", 0)
        ptype = item.get("type", "")
        grade = item.get("grade", "")
        stock_label = f"库存 {stock} 颗" if stock <= 5 else f"库存 {stock} 颗"
        lines.append(f"  {i}. {name}（{ptype}）¥{price}/颗 · {stock_label}")
        if grade:
            lines.append(f"     {grade}")
    return "\n".join(lines)
