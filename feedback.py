"""采纳率追踪 — 线程安全的日志记录 + 统计"""
import json
import threading
from pathlib import Path
from datetime import datetime

FEEDBACK_LOG = Path(__file__).parent / "feedback_log.jsonl"
_lock = threading.Lock()


def log_feedback(action: str, question: str = "", reply: str = "") -> dict:
    """记录用户行为（线程安全）。action: 'copied' 或 'regenerated'"""
    entry = {
        "action": action,
        "question": question[:200],
        "reply_preview": reply[:200],
        "timestamp": datetime.now().isoformat(),
    }
    with _lock:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return get_adoption_stats()


def get_adoption_stats() -> dict:
    """统计采纳率（线程安全）"""
    if not FEEDBACK_LOG.exists():
        return {"total": 0, "copied": 0, "regenerated": 0, "adoption_rate": 0.0}

    actions = []
    with _lock:
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    actions.append(entry["action"])
                except Exception:
                    pass

    total = len(actions)
    copied = actions.count("copied")
    return {
        "total": total,
        "copied": copied,
        "regenerated": total - copied,
        "adoption_rate": round(copied / total * 100, 1) if total > 0 else 0.0,
    }
