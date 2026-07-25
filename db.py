"""
db.py — PostgreSQL 持久化（自动降级到 JSON 文件）

用法：
    from db import db
    db.save_conversation("mom", "问题", "回复")
    history = db.get_conversations("mom")

PostgreSQL 不可用时自动使用 JSON 文件，无需改任何代码。
"""

import os
import json
import threading
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://pearl:pearl@localhost:5432/pearl_assistant")

MEMORY_FILE = Path(__file__).parent / "conversation_memory.json"
FEEDBACK_FILE = Path(__file__).parent / "feedback_log.jsonl"

_conn_pool = None
_lock = threading.Lock()


def _try_connect_pg():
    """尝试连接 PostgreSQL。失败返回 None。"""
    global _conn_pool
    if _conn_pool is not None:
        return _conn_pool
    try:
        import psycopg2
        import psycopg2.pool
        _conn_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=3, dsn=DATABASE_URL, connect_timeout=3
        )
        # 建表（幂等）
        conn = _conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(50) NOT NULL,
                        question TEXT NOT NULL,
                        reply TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, created_at);
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id SERIAL PRIMARY KEY,
                        action VARCHAR(20) NOT NULL,
                        question TEXT DEFAULT '',
                        reply TEXT DEFAULT '',
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
            conn.commit()
        finally:
            _conn_pool.putconn(conn)
        print(f"[db] PostgreSQL 连接成功 — {DATABASE_URL}")
        return _conn_pool
    except Exception:
        print(f"[db] PostgreSQL 不可用，使用 JSON 文件存储")
        _conn_pool = None
        return None


class Database:
    """统一持久化接口。PostgreSQL 优先，JSON 文件兜底。"""

    # ---- 对话记忆 ----
    def get_conversations(self, session_id: str, limit: int = 15) -> list[dict]:
        pool = _try_connect_pg()
        if pool:
            try:
                conn = pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT question, reply FROM conversations "
                            "WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
                            (session_id, limit),
                        )
                        rows = cur.fetchall()
                    return [{"question": r[0], "reply": r[1]} for r in reversed(rows)]
                finally:
                    pool.putconn(conn)
            except Exception:
                pass

        # 回退到 JSON 文件
        return self._json_get_conversations(session_id, limit)

    def save_conversation(self, session_id: str, question: str, reply: str):
        pool = _try_connect_pg()
        if pool:
            try:
                conn = pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO conversations (session_id, question, reply) VALUES (%s, %s, %s)",
                            (session_id, question, reply),
                        )
                        # 保留最近 15 条，删旧的
                        cur.execute(
                            "DELETE FROM conversations WHERE session_id = %s AND id NOT IN "
                            "(SELECT id FROM conversations WHERE session_id = %s ORDER BY created_at DESC LIMIT 15)",
                            (session_id, session_id),
                        )
                    conn.commit()
                    return
                finally:
                    pool.putconn(conn)
            except Exception:
                pass

        self._json_save_conversation(session_id, question, reply)

    # ---- JSON 回退（保留原有逻辑） ----
    @staticmethod
    def _json_get_conversations(session_id: str, limit: int) -> list[dict]:
        if not MEMORY_FILE.exists():
            return []
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            msgs = data.get(session_id, [])
            return msgs[-limit:] if isinstance(msgs, list) else []
        except Exception:
            return []

    @staticmethod
    def _json_save_conversation(session_id: str, question: str, reply: str):
        data = {}
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        msgs = data.get(session_id, [])
        if not isinstance(msgs, list):
            msgs = []
        msgs.append({"question": question, "reply": reply})
        data[session_id] = msgs[-15:]
        with _lock:
            MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 反馈日志 ----
    def log_feedback(self, action: str, question: str = "", reply: str = ""):
        pool = _try_connect_pg()
        if pool:
            try:
                conn = pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO feedback (action, question, reply) VALUES (%s, %s, %s)",
                            (action, question, reply),
                        )
                    conn.commit()
                    return
                finally:
                    pool.putconn(conn)
            except Exception:
                pass

        try:
            with _lock:
                with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "action": action, "question": question, "reply": reply,
                    }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_feedback_stats(self) -> dict:
        pool = _try_connect_pg()
        if pool:
            try:
                conn = pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM feedback")
                        total = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM feedback WHERE action = 'copied'")
                        copied = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM feedback WHERE action = 'regenerated'")
                        regenerated = cur.fetchone()[0]
                    return {
                        "total": total,
                        "copied": copied,
                        "regenerated": regenerated,
                        "adoption_rate": round(copied / max(total, 1), 2),
                    }
                finally:
                    pool.putconn(conn)
            except Exception:
                pass

        return self._json_get_feedback_stats()

    @staticmethod
    def _json_get_feedback_stats() -> dict:
        if not FEEDBACK_FILE.exists():
            return {"total": 0, "copied": 0, "regenerated": 0, "adoption_rate": 0.0}
        total, copied, regenerated = 0, 0, 0
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        total += 1
                        if entry.get("action") == "copied":
                            copied += 1
                        elif entry.get("action") == "regenerated":
                            regenerated += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return {
            "total": total,
            "copied": copied,
            "regenerated": regenerated,
            "adoption_rate": round(copied / max(total, 1), 2),
        }


db = Database()
