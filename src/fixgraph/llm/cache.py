"""SQLite response cache keyed on (model, prompt, params). Makes long jobs resumable.

`CachedLLMClient` wraps any LLMClient; llm.factory applies it by default (`use_cache=True`).
Uses: llm.base.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from types import TracebackType
from typing import Self

from fixgraph.llm.base import LLMClient, LLMRequest, LLMResponse


def cache_key(request: LLMRequest) -> str:
    payload = json.dumps(request.cache_payload(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Callers may run 1-2 concurrent requests from a thread pool; one lock serializes access.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else LLMResponse.model_validate_json(row[0])

    def put(self, key: str, response: LLMResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, response) VALUES (?, ?)",
                (key, response.model_dump_json()),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class CachedLLMClient:
    """Wraps any LLMClient; identical requests are served from the cache. Counts hits and
    misses (thread-safe) so runs and the API can report their cache hit rate."""

    def __init__(self, inner: LLMClient, cache: SQLiteCache) -> None:
        self.inner = inner
        self.cache = cache
        self.hits = 0
        self.misses = 0
        self._count_lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = cache_key(request)
        hit = self.cache.get(key)
        if hit is not None:
            with self._count_lock:
                self.hits += 1
            return hit.model_copy(update={"cached": True})
        response = self.inner.complete(request)
        self.cache.put(key, response)
        with self._count_lock:
            self.misses += 1
        return response

    def stats(self) -> dict[str, float]:
        with self._count_lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }

    def close(self) -> None:
        self.inner.close()
        self.cache.close()
