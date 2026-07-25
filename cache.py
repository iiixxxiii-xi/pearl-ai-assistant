"""
cache.py — Redis 缓存（自动降级到内存字典）

用法：
    from cache import cache
    cache.set("key", "value", ttl=300)
    value = cache.get("key")

Redis 不可用时自动使用内存字典，无需改任何代码。
"""

import os
import threading
import time

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

_redis_client = None
_memory_fallback: dict[str, tuple[str, float]] = {}
_memory_lock = threading.Lock()


def _try_connect_redis():
    """尝试连接 Redis。失败返回 None，不抛异常。"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, decode_responses=True)
        client.ping()
        _redis_client = client
        print(f"[cache] Redis 连接成功 — {REDIS_URL}")
        return client
    except Exception:
        print(f"[cache] Redis 不可用，使用内存缓存")
        return None


class Cache:
    """统一缓存接口。Redis 优先，内存字典兜底。"""

    def get(self, key: str) -> str | None:
        client = _try_connect_redis()
        if client:
            try:
                return client.get(key)
            except Exception:
                pass

        with _memory_lock:
            entry = _memory_fallback.get(key)
            if entry:
                value, expires_at = entry
                if time.time() < expires_at:
                    return value
                del _memory_fallback[key]
        return None

    def set(self, key: str, value: str, ttl: int = 300):
        client = _try_connect_redis()
        if client:
            try:
                client.setex(key, ttl, value)
                return
            except Exception:
                pass

        with _memory_lock:
            _memory_fallback[key] = (value, time.time() + ttl)


# 模块级单例
cache = Cache()
